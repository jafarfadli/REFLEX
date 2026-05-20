import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import mediapipe as mp

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── Detection config ──────────────────────────────────────────────────────────
ALERT_THRESHOLD       = 0.65

# Fusion weights
W_EYE                 = 0.35
W_PERCLOS             = 0.15
W_YAWN                = 0.20
W_NOD                 = 0.30

# Eye closure (EAR-based)
EYE_MIN_DURATION      = 0.4    # sustained closed ≥ this → microsleep
PERCLOS_WINDOW_SEC    = 60
PERCLOS_THRESHOLD     = 0.15   # 15% drowsy threshold (industry standard)
EAR_FALLBACK_THRESHOLD = 0.25  # used if --skip-calib

# Calibration
CALIB_OPEN_SEC         = 5.0   # seconds capturing OPEN baseline (also pitch)
CALIB_CLOSED_SEC       = 5.0   # seconds capturing CLOSED baseline
CALIB_COUNTDOWN_SEC    = 3
CALIB_MIN_SAMPLES      = 20
CALIB_WARN_SPREAD      = 0.04  # warn if open-closed spread is below this

# Yawn
YAWN_CONF_THRESHOLD   = 0.70
YAWN_MIN_DURATION     = 1.0

# Nod
NOD_THRESHOLD_DEG     = 12.0
NOD_MIN_DURATION      = 0.8

ROLLING_WINDOW_SEC    = 30
FACE_CONF_MIN         = 0.40
MOUTH_CROP_START      = 0.55
IMG_SIZE              = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Color palette (BGR) ───────────────────────────────────────────────────────
COLOR_OK      = (40, 200, 80)
COLOR_WARNING = (0, 165, 255)
COLOR_ALERT   = (0, 0, 220)
COLOR_HUD     = (200, 200, 200)
COLOR_OPEN    = (0, 255, 0)
COLOR_CLOSED  = (0, 100, 255)

WINDOW_NAME = "Fatigue Detection — V2X Prototype"

# ── Head pose: 3D model + landmark indices ────────────────────────────────────
FACE_3D_MODEL = np.array([
    [   0.0,   0.0,   0.0],
    [   0.0, -63.6, -12.5],
    [ -43.3,  32.7, -26.0],
    [  43.3,  32.7, -26.0],
    [ -28.9, -28.9, -24.1],
    [  28.9, -28.9, -24.1],
], dtype=np.float64)
MP_POSE_IDS = [1, 152, 33, 263, 61, 291]

# ── EAR landmarks ─────────────────────────────────────────────────────────────
# Order: [outer_corner, upper1, upper2, inner_corner, lower1, lower2]
LEFT_EYE_EAR  = [33,  159, 158, 133, 153, 145]
RIGHT_EYE_EAR = [263, 386, 385, 362, 380, 373]


# ── EAR helpers ───────────────────────────────────────────────────────────────

def _single_eye_ear(landmarks, ids, fw, fh) -> float:
    pts = np.array(
        [[landmarks[i].x * fw, landmarks[i].y * fh] for i in ids],
        dtype=np.float64,
    )
    vert  = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return float(vert / (2.0 * horiz)) if horiz > 1e-6 else 0.0


def compute_ear(landmarks, fw: int, fh: int) -> float:
    """Average EAR across both eyes."""
    l = _single_eye_ear(landmarks, LEFT_EYE_EAR,  fw, fh)
    r = _single_eye_ear(landmarks, RIGHT_EYE_EAR, fw, fh)
    return (l + r) / 2.0


# ── CNN helpers (yawn only) ───────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_cnn(model_path: Path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    classes = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(classes)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    return model, classes


@torch.no_grad()
def classify(model, classes: list, crop: np.ndarray, transform) -> tuple[str, float]:
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "unknown", 0.0
    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    idx    = probs.argmax().item()
    return classes[idx], probs[idx].item()


def positive_prob(label: str, conf: float, positive_class: str) -> float:
    return conf if label == positive_class else 1.0 - conf


# ── Frame helpers ─────────────────────────────────────────────────────────────

def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def mouth_crop(face_bgr: np.ndarray) -> np.ndarray | None:
    if face_bgr is None or face_bgr.size == 0:
        return None
    fh = face_bgr.shape[0]
    return face_bgr[int(fh * MOUTH_CROP_START):, :]


def estimate_pitch(landmarks, fw: int, fh: int) -> float | None:
    image_pts = np.array(
        [[landmarks[i].x * fw, landmarks[i].y * fh] for i in MP_POSE_IDS],
        dtype=np.float64,
    )
    focal = float(fw)
    cam_matrix = np.array([
        [focal,   0.0, fw / 2.0],
        [  0.0, focal, fh / 2.0],
        [  0.0,   0.0,      1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL, image_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy < 1e-6:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
    else:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
    pitch_deg = float(np.degrees(pitch))
    if pitch_deg > 90:
        pitch_deg -= 180
    elif pitch_deg < -90:
        pitch_deg += 180
    return pitch_deg


# ── Calibration phase ─────────────────────────────────────────────────────────

def _show_countdown(cap, prompt: str, color: tuple, seconds: int) -> bool:
    """Returns False if user pressed q."""
    for sec in range(seconds, 0, -1):
        end_t = time.time() + 1.0
        while time.time() < end_t:
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
            cv2.putText(frame, f"Get ready: {prompt}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Starting in {sec}...", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                return False
    return True


def _capture_phase(cap, face_mesh, prompt: str, color: tuple,
                   duration: float, want_pitch: bool):
    """Capture EAR (and optionally pitch) values over the duration."""
    ear_values   = []
    pitch_values = []
    start_t      = time.time()

    while time.time() - start_t < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ear_now = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear_now = compute_ear(lm, w, h)
            ear_values.append(ear_now)
            if want_pitch:
                p = estimate_pitch(lm, w, h)
                if p is not None:
                    pitch_values.append(p)

        # Overlay
        elapsed   = time.time() - start_t
        remaining = duration - elapsed
        cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"RECORDING: {prompt}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        ear_str = f"EAR = {ear_now:.4f}" if ear_now is not None else "no face"
        cv2.putText(frame,
                    f"{remaining:4.1f}s left   |   {ear_str}   |   samples = {len(ear_values)}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # Progress bar
        bar_w = int(w * (elapsed / duration))
        cv2.rectangle(frame, (0, 130), (bar_w, 136), color, -1)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return ear_values, pitch_values


def run_calibration(cap, face_mesh) -> tuple[float, float, float] | None:
    """
    Runs two-phase calibration. Returns:
      (ear_threshold, baseline_open_ear, baseline_pitch)
    or None if aborted/failed.
    """
    print("\n  ── CALIBRATION ────────────────────────────────────────")
    print("     Two phases, ~10 seconds total.\n")

    # Phase 1: OPEN eyes (also gather pitch baseline)
    if not _show_countdown(cap, "look at camera with eyes OPEN",
                           COLOR_OPEN, CALIB_COUNTDOWN_SEC):
        return None
    print("     Phase 1/2: capturing OPEN baseline (eyes + pitch)...")
    open_ears, pitch_vals = _capture_phase(
        cap, face_mesh,
        prompt="OPEN your eyes normally",
        color=COLOR_OPEN,
        duration=CALIB_OPEN_SEC,
        want_pitch=True,
    )

    # Phase 2: CLOSED eyes
    if not _show_countdown(cap, "CLOSE your eyes fully",
                           COLOR_CLOSED, CALIB_COUNTDOWN_SEC):
        return None
    print("     Phase 2/2: capturing CLOSED baseline...")
    closed_ears, _ = _capture_phase(
        cap, face_mesh,
        prompt="CLOSE your eyes fully (keep them shut)",
        color=COLOR_CLOSED,
        duration=CALIB_CLOSED_SEC,
        want_pitch=False,
    )

    # Sanity
    if len(open_ears) < CALIB_MIN_SAMPLES or len(closed_ears) < CALIB_MIN_SAMPLES:
        print(f"     [ERROR] Not enough samples (open={len(open_ears)}, closed={len(closed_ears)}).")
        return None
    if len(pitch_vals) < CALIB_MIN_SAMPLES:
        print(f"     [WARN] Few pitch samples ({len(pitch_vals)}). Nod detection may be unstable.")

    # Compute thresholds
    open_med   = float(np.median(open_ears))
    closed_med = float(np.median(closed_ears))
    spread     = open_med - closed_med
    threshold  = (open_med + closed_med) / 2.0
    pitch_med  = float(np.median(pitch_vals)) if pitch_vals else 0.0

    print(f"\n     OPEN  EAR median   : {open_med:.4f}  (n={len(open_ears)})")
    print(f"     CLOSED EAR median  : {closed_med:.4f}  (n={len(closed_ears)})")
    print(f"     Spread             : {spread:.4f}")
    print(f"     EAR threshold      : {threshold:.4f}")
    print(f"     Pitch baseline     : {pitch_med:+.2f}°")

    if spread < CALIB_WARN_SPREAD:
        print(f"     ⚠  Spread is very small. Eye detection may misfire.")

    return threshold, open_med, pitch_med


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_overlay(frame, bbox,
                 ear_val, ear_threshold, is_eyes_closed,
                 yawn_label, yawn_conf, score,
                 yawn_count, nod_count, closure_count,
                 perclos, pitch_str, alert):
    x1, y1, x2, y2 = bbox

    if alert:
        color = COLOR_ALERT
    elif score > 0.45:
        color = COLOR_WARNING
    else:
        color = COLOR_OK

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    bar_w  = x2 - x1
    filled = int(bar_w * min(score, 1.0))
    cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 8), (60, 60, 60), -1)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 8), color, -1)

    eye_state = "CLOSED" if is_eyes_closed else "OPEN"
    ear_str   = f"{ear_val:.3f}" if ear_val is not None else "—"
    lines = [
        f"Eyes  : {eye_state}    EAR {ear_str} / thr {ear_threshold:.3f}",
        f"Yawn  : {yawn_label}  {yawn_conf:.2f}",
        f"Pitch : {pitch_str}",
        f"Score : {score:.3f}   PERCLOS: {perclos*100:.1f}%",
        f"Events: Y/30s {yawn_count}   N/30s {nod_count}   C/30s {closure_count}",
    ]
    for i, line in enumerate(reversed(lines)):
        y_pos = y1 - 8 - i * 18
        if y_pos < 15:
            y_pos = y2 + 22 + i * 18
        cv2.putText(frame, line, (x1, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if alert:
        h_fr = frame.shape[0]
        cv2.rectangle(frame, (0, h_fr - 40), (frame.shape[1], h_fr), (0, 0, 180), -1)
        cv2.putText(frame, "  !! FATIGUE ALERT — V2X WARNING BROADCAST !!",
                    (10, h_fr - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fatigue Detection — V2X Prototype")
    parser.add_argument("--camera",        type=int,   default=0,                help="Webcam index")
    parser.add_argument("--threshold",     type=float, default=ALERT_THRESHOLD,  help="Alert threshold [0-1]")
    parser.add_argument("--no-display",    action="store_true",                  help="Disable OpenCV window")
    parser.add_argument("--skip-calib",    action="store_true",                  help="Skip calibration (use fallback EAR threshold)")
    parser.add_argument("--ear-threshold", type=float, default=EAR_FALLBACK_THRESHOLD,
                        help="EAR threshold when --skip-calib (default: %(default)s)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Fatigue Detection System  |  V2X Prototype")
    print(f"{'='*60}")
    print(f"  Device          : {DEVICE}")
    print(f"  Alert threshold : {args.threshold}")

    # ── Load yawn CNN ─────────────────────────────────────────────────────────
    yaw_path = MODELS_DIR / "yawn_clf.pt"
    if not yaw_path.exists():
        print(f"\n  [ERROR] yawn_clf model not found: {yaw_path}")
        print("          Run:  python train_cnn.py --clf yawn_clf  first.\n")
        sys.exit(1)

    print("  Loading yawn CNN...")
    yaw_model, yaw_cls = load_cnn(yaw_path)
    print(f"    yawn_clf classes: {yaw_cls}")

    # ── YOLO ──────────────────────────────────────────────────────────────────
    print("  Loading YOLO face detector...")
    try:
        from ultralytics import YOLO
        yolo_path = MODELS_DIR / "yolo26n-face.pt"
        if not yolo_path.exists():
            print(f"\n  [ERROR] YOLO weights not found: {yolo_path}\n")
            sys.exit(1)
        yolo = YOLO(str(yolo_path))
    except Exception as e:
        print(f"\n  [ERROR] Failed to load YOLO: {e}\n")
        sys.exit(1)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    print("  Loading MediaPipe Face Mesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    transform = get_transform()

    # ── Webcam ────────────────────────────────────────────────────────────────
    print(f"\n  Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open camera index {args.camera}")
        sys.exit(1)

    # ── Calibration ───────────────────────────────────────────────────────────
    if args.skip_calib:
        ear_threshold     = args.ear_threshold
        baseline_ear_open = ear_threshold / 0.5    # rough approximation
        baseline_pitch    = 0.0
        print(f"\n  [skip-calib]  EAR threshold = {ear_threshold:.3f}, pitch baseline = 0°\n")
    else:
        calib = run_calibration(cap, face_mesh)
        if calib is None:
            print("\n  Calibration aborted. Exiting.\n")
            cap.release()
            sys.exit(1)
        ear_threshold, baseline_ear_open, baseline_pitch = calib

    print("\n  Detection started. Press  q  to quit.\n")

    # ── State ─────────────────────────────────────────────────────────────────
    # Eye closure (sustained → microsleep)
    closure_streak_start:    float | None = None
    closure_already_counted: bool         = False
    closure_events:          deque        = deque()
    perclos_history:         deque        = deque()

    # Yawn
    yawn_streak_start:    float | None = None
    yawn_already_counted: bool         = False
    yawn_times:           deque        = deque()

    # Nod
    nod_streak_start:    float | None = None
    nod_already_counted: bool         = False
    nod_times:           deque        = deque()

    # FPS
    fps_counter, fps_t0, fps = 0, time.time(), 0.0

    # Header
    print(f"  {'Status':<11} {'Score':>6}  {'EAR':>14}  {'Yawn':>14}  "
          f"{'Pitch':>7}  {'PCL':>5}  {'Y/30':>4}  {'N/30':>4}  {'C/30':>4}  {'FPS':>5}")
    print("  " + "-" * 95)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n  [WARN] Empty frame — retrying...")
                time.sleep(0.05)
                continue

            now = time.time()
            fps_counter += 1
            if fps_counter >= 15:
                fps         = fps_counter / (now - fps_t0)
                fps_counter = 0
                fps_t0      = now

            # Prune rolling windows
            while yawn_times      and now - yawn_times[0]         > ROLLING_WINDOW_SEC:
                yawn_times.popleft()
            while nod_times       and now - nod_times[0]          > ROLLING_WINDOW_SEC:
                nod_times.popleft()
            while closure_events  and now - closure_events[0]     > ROLLING_WINDOW_SEC:
                closure_events.popleft()
            while perclos_history and now - perclos_history[0][0] > PERCLOS_WINDOW_SEC:
                perclos_history.popleft()

            # ── YOLO face detection ───────────────────────────────────────────
            results    = yolo(frame, verbose=False, conf=FACE_CONF_MIN, imgsz=640)
            detections = results[0].boxes

            if detections is None or len(detections) == 0:
                print(f"\r  {'[no face]':<11} {'—':>6}  {'—':>14}  {'—':>14}  "
                      f"{'—':>7}  {'—':>5}  {'—':>4}  {'—':>4}  {'—':>4}  {fps:>4.1f}fps",
                      end="", flush=True)
                if not args.no_display:
                    cv2.putText(frame, f"No face  FPS:{fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1)
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Largest face
            boxes = detections.xyxy.cpu().numpy()
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best  = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[best])
            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)

            face_crop = frame[y1:y2, x1:x2]
            m_crop    = mouth_crop(face_crop)

            # ── MediaPipe (EAR + head pose) ───────────────────────────────────
            mp_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ear_val: float | None       = None
            current_pitch: float | None = None
            if mp_results.multi_face_landmarks:
                lm = mp_results.multi_face_landmarks[0].landmark
                ear_val       = compute_ear(lm, fw, fh)
                current_pitch = estimate_pitch(lm, fw, fh)

            # ── Eye closure state (binary, from threshold) ────────────────────
            is_eyes_closed = (ear_val is not None and ear_val < ear_threshold)

            # ── Sustained closure (microsleep) ────────────────────────────────
            if is_eyes_closed:
                if closure_streak_start is None:
                    closure_streak_start    = now
                    closure_already_counted = False
                if (not closure_already_counted
                        and now - closure_streak_start >= EYE_MIN_DURATION):
                    closure_events.append(now)
                    closure_already_counted = True
            else:
                closure_streak_start    = None
                closure_already_counted = False
            closure_count = len(closure_events)

            # ── PERCLOS ───────────────────────────────────────────────────────
            if ear_val is not None:
                perclos_history.append((now, is_eyes_closed))
            perclos = (sum(1 for _, c in perclos_history if c) / len(perclos_history)
                       if perclos_history else 0.0)

            # ── Eye signal for fusion (smooth interpolation) ──────────────────
            # 0.0 when ear == baseline_open, 1.0 when ear == ear_threshold or below.
            # Linear ramp between, clamped.
            if ear_val is None:
                eye_signal = 0.0
            elif baseline_ear_open > ear_threshold:
                eye_signal = float(np.clip(
                    (baseline_ear_open - ear_val) / (baseline_ear_open - ear_threshold),
                    0.0, 1.0,
                ))
            else:
                eye_signal = 1.0 if is_eyes_closed else 0.0

            # ── Yawn ──────────────────────────────────────────────────────────
            yaw_label, yaw_conf = classify(yaw_model, yaw_cls, m_crop, transform)
            yaw_prob = positive_prob(yaw_label, yaw_conf, "yawn")

            is_mouth_open = yaw_label == "yawn" and yaw_conf >= YAWN_CONF_THRESHOLD
            if is_mouth_open:
                if yawn_streak_start is None:
                    yawn_streak_start    = now
                    yawn_already_counted = False
                if (not yawn_already_counted
                        and now - yawn_streak_start >= YAWN_MIN_DURATION):
                    yawn_times.append(now)
                    yawn_already_counted = True
            else:
                yawn_streak_start    = None
                yawn_already_counted = False
            yawn_count = len(yawn_times)

            # ── Nod detection ─────────────────────────────────────────────────
            nod_intensity = 0.0
            is_head_down  = False
            pitch_offset  = 0.0
            if current_pitch is not None:
                pitch_offset  = current_pitch - baseline_pitch
                nod_intensity = min(abs(pitch_offset) / NOD_THRESHOLD_DEG, 1.0)
                is_head_down  = abs(pitch_offset) >= NOD_THRESHOLD_DEG

            if is_head_down:
                if nod_streak_start is None:
                    nod_streak_start    = now
                    nod_already_counted = False
                if (not nod_already_counted
                        and now - nod_streak_start >= NOD_MIN_DURATION):
                    nod_times.append(now)
                    nod_already_counted = True
            else:
                nod_streak_start    = None
                nod_already_counted = False
            nod_count = len(nod_times)

            # ── Fusion ────────────────────────────────────────────────────────
            perclos_signal = min(perclos / PERCLOS_THRESHOLD, 1.0)
            score = (W_EYE     * eye_signal
                     + W_PERCLOS * perclos_signal
                     + W_YAWN    * yaw_prob
                     + W_NOD     * nod_intensity)
            alert = score >= args.threshold

            # ── Terminal output ───────────────────────────────────────────────
            pitch_str = f"{pitch_offset:+5.1f}°"
            ear_str   = f"{ear_val:.3f}" if ear_val is not None else "—"
            eye_disp  = "CLOSED" if is_eyes_closed else "OPEN"

            status = "🚨 ALERT  " if alert else "✅ OK     "
            print(
                f"\r  {status:<11} {score:>6.3f}  "
                f"{ear_str:>5}/{eye_disp:>6}  "
                f"{yaw_label:>8}({yaw_conf:.2f})  "
                f"{pitch_str:>7}  "
                f"{perclos*100:>4.1f}%  "
                f"{yawn_count:>4d}  {nod_count:>4d}  {closure_count:>4d}  "
                f"{fps:>4.1f}fps",
                end="", flush=True,
            )

            if alert:
                ts = time.strftime("%H:%M:%S")
                print(
                    f"\n  ┌─ V2X EVENT [{ts}] ────────────────────────────────────\n"
                    f"  │  fatigue_score : {score:.4f}\n"
                    f"  │  eye_signal    : {eye_signal:.4f}    "
                    f"(EAR={ear_str}, threshold={ear_threshold:.3f}, closed={is_eyes_closed})\n"
                    f"  │  perclos       : {perclos*100:.1f}%   "
                    f"(threshold={PERCLOS_THRESHOLD*100:.0f}%)\n"
                    f"  │  yawn_conf     : {yaw_label} @ {yaw_conf:.4f}\n"
                    f"  │  pitch_offset  : {pitch_offset:+.2f}°  "
                    f"(baseline={baseline_pitch:+.2f}°)\n"
                    f"  │  events/{ROLLING_WINDOW_SEC}s   : "
                    f"closures={closure_count}  yawns={yawn_count}  nods={nod_count}\n"
                    f"  └─ ACTION        : BROADCAST WARNING TO SURROUNDING UNITS"
                )

            # ── Display ───────────────────────────────────────────────────────
            if not args.no_display:
                draw_overlay(frame, (x1, y1, x2, y2),
                             ear_val, ear_threshold, is_eyes_closed,
                             yaw_label, yaw_conf,
                             score, yawn_count, nod_count, closure_count,
                             perclos, pitch_str, alert)
                cv2.putText(frame, f"FPS: {fps:.1f}", (fw - 100, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD, 1)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    finally:
        cap.release()
        face_mesh.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("\n  Camera released. Bye!\n")


if __name__ == "__main__":
    main()