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

# Eye closure (CNN)
EYE_CLOSURE_CONF      = 0.70
EYE_MIN_DURATION      = 0.4
PERCLOS_WINDOW_SEC    = 60
PERCLOS_THRESHOLD     = 0.15
EYE_PAD_RATIO         = 0.4

# Yawn
YAWN_CONF_THRESHOLD   = 0.70
YAWN_MIN_DURATION     = 1.0

# Nod
NOD_THRESHOLD_DEG     = 12.0
NOD_MIN_DURATION      = 0.8
CALIBRATION_FRAMES    = 30

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

# ── Head pose ─────────────────────────────────────────────────────────────────
FACE_3D_MODEL = np.array([
    [   0.0,   0.0,   0.0],
    [   0.0, -63.6, -12.5],
    [ -43.3,  32.7, -26.0],
    [  43.3,  32.7, -26.0],
    [ -28.9, -28.9, -24.1],
    [  28.9, -28.9, -24.1],
], dtype=np.float64)
MP_POSE_IDS = [1, 152, 33, 263, 61, 291]

# ── Eye contour landmarks (MUST MATCH record_eyes.py) ────────────────────────
LEFT_EYE_IDS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_IDS = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466,
]


# ── Eye crop function (MUST match record_eyes.py byte-for-byte) ───────────────

def extract_eye_crops(frame, landmarks, frame_w, frame_h,
                      pad_ratio: float = EYE_PAD_RATIO) -> list[np.ndarray]:
    """Square eye crops with proportional padding."""
    eyes = []
    for ids in (LEFT_EYE_IDS, RIGHT_EYE_IDS):
        xs = [landmarks[i].x * frame_w for i in ids]
        ys = [landmarks[i].y * frame_h for i in ids]

        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * (1.0 + pad_ratio) / 2.0

        x1 = int(max(0,        cx - half))
        y1 = int(max(0,        cy - half))
        x2 = int(min(frame_w,  cx + half))
        y2 = int(min(frame_h,  cy + half))

        if x2 - x1 >= 15 and y2 - y1 >= 15:
            eyes.append(frame[y1:y2, x1:x2])
    return eyes


# ── CNN helpers ───────────────────────────────────────────────────────────────

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


# ── Crop helpers ──────────────────────────────────────────────────────────────

def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def mouth_crop(face_bgr: np.ndarray) -> np.ndarray | None:
    if face_bgr is None or face_bgr.size == 0:
        return None
    fh = face_bgr.shape[0]
    return face_bgr[int(fh * MOUTH_CROP_START) :, :]


# ── Head pose ─────────────────────────────────────────────────────────────────

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


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_overlay(frame, bbox, eye_label, eye_conf,
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

    lines = [
        f"Eyes  : {eye_label}  {eye_conf:.2f}",
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
        cv2.putText(frame, "  !! WOI BANGUN WOI !!",
                    (10, h_fr - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fatigue Detection — V2X Prototype")
    parser.add_argument("--camera",     type=int,   default=0,                help="Webcam index")
    parser.add_argument("--threshold",  type=float, default=ALERT_THRESHOLD,  help="Alert threshold [0-1]")
    parser.add_argument("--no-display", action="store_true",                  help="Disable OpenCV window")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Fatigue Detection System  |  V2X Prototype")
    print(f"{'='*60}")
    print(f"  Device     : {DEVICE}")
    print(f"  Threshold  : {args.threshold}")

    # ── Load CNNs ─────────────────────────────────────────────────────────────
    for name, path in [("eye_clf",  MODELS_DIR / "eye_clf.pt"),
                       ("yawn_clf", MODELS_DIR / "yawn_clf.pt")]:
        if not path.exists():
            print(f"\n  [ERROR] {name} model not found: {path}")
            print("          Run:  python train_cnn.py  first.\n")
            sys.exit(1)

    print("  Loading CNN classifiers...")
    eye_model, eye_cls = load_cnn(MODELS_DIR / "eye_clf.pt")
    yaw_model, yaw_cls = load_cnn(MODELS_DIR / "yawn_clf.pt")
    print(f"    eye_clf   classes: {eye_cls}")
    print(f"    yawn_clf  classes: {yaw_cls}")

    # ── YOLO ──────────────────────────────────────────────────────────────────
    print("  Loading YOLO face detector...")
    try:
        from ultralytics import YOLO
        yolo_path = BASE_DIR / "models" / "yolo26n-face.pt"
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
    print(f"\n  Opening camera {args.camera}...  Press  q  to quit.")
    print("  >>> CALIBRATION: please look straight forward for ~2 seconds <<<\n")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open camera index {args.camera}")
        sys.exit(1)

    # ── State ─────────────────────────────────────────────────────────────────
    closure_streak_start:    float | None = None
    closure_already_counted: bool         = False
    closure_events:          deque        = deque()
    perclos_history:         deque        = deque()

    yawn_streak_start:    float | None = None
    yawn_already_counted: bool         = False
    yawn_times:           deque        = deque()

    pitch_buffer:        list[float]  = []
    baseline_pitch:      float | None = None
    nod_streak_start:    float | None = None
    nod_already_counted: bool         = False
    nod_times:           deque        = deque()

    fps_counter, fps_t0, fps = 0, time.time(), 0.0

    print(f"  {'Status':<11} {'Score':>6}  {'Eyes':>14}  {'Yawn':>14}  "
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

            while yawn_times      and now - yawn_times[0]         > ROLLING_WINDOW_SEC:
                yawn_times.popleft()
            while nod_times       and now - nod_times[0]          > ROLLING_WINDOW_SEC:
                nod_times.popleft()
            while closure_events  and now - closure_events[0]     > ROLLING_WINDOW_SEC:
                closure_events.popleft()
            while perclos_history and now - perclos_history[0][0] > PERCLOS_WINDOW_SEC:
                perclos_history.popleft()

            # YOLO
            results    = yolo(frame, verbose=False, conf=FACE_CONF_MIN, imgsz=640)
            detections = results[0].boxes

            if detections is None or len(detections) == 0:
                print(f"\r  {'[no face]':<11} {'—':>6}  {'—':>14}  {'—':>14}  "
                      f"{'—':>7}  {'—':>5}  {'—':>4}  {'—':>4}  {'—':>4}  {fps:>4.1f}fps",
                      end="", flush=True)
                if not args.no_display:
                    cv2.putText(frame, f"No face  FPS:{fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1)
                    cv2.imshow("REFLEX", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            boxes = detections.xyxy.cpu().numpy()
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best  = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[best])
            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)

            face_crop = frame[y1:y2, x1:x2]
            m_crop    = mouth_crop(face_crop)

            # MediaPipe
            mp_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_pitch: float | None = None
            eye_crops: list = []
            if mp_results.multi_face_landmarks:
                lm = mp_results.multi_face_landmarks[0].landmark
                current_pitch = estimate_pitch(lm, fw, fh)
                eye_crops     = extract_eye_crops(frame, lm, fw, fh)

            # Eye CNN
            eye_closure_prob: float | None = None
            eye_label_disp = "—"
            eye_conf_disp  = 0.0
            if eye_crops:
                closure_probs = []
                for crop in eye_crops:
                    label, conf = classify(eye_model, eye_cls, crop, transform)
                    if label == "unknown":
                        continue
                    closure_probs.append(positive_prob(label, conf, "closed"))
                if closure_probs:
                    eye_closure_prob = float(np.mean(closure_probs))
                    eye_label_disp   = "closed" if eye_closure_prob >= 0.5 else "open"
                    eye_conf_disp    = (eye_closure_prob if eye_label_disp == "closed"
                                        else 1.0 - eye_closure_prob)

            # Sustained closure
            is_eyes_closed = (eye_closure_prob is not None
                              and eye_closure_prob >= EYE_CLOSURE_CONF)
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

            # PERCLOS
            if eye_closure_prob is not None:
                perclos_history.append((now, is_eyes_closed))
            if perclos_history:
                perclos = sum(1 for _, c in perclos_history if c) / len(perclos_history)
            else:
                perclos = 0.0

            # Yawn
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

            # Head-pose calibration + nod
            if current_pitch is not None and baseline_pitch is None:
                pitch_buffer.append(current_pitch)
                if len(pitch_buffer) >= CALIBRATION_FRAMES:
                    baseline_pitch = float(np.median(pitch_buffer))
                    print(f"\n  [calibrated]  baseline pitch = {baseline_pitch:+.2f}°\n")

            nod_intensity = 0.0
            is_head_down  = False
            pitch_offset  = 0.0
            if current_pitch is not None and baseline_pitch is not None:
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

            # Fusion
            eye_signal     = eye_closure_prob if eye_closure_prob is not None else 0.0
            perclos_signal = min(perclos / PERCLOS_THRESHOLD, 1.0)

            score = (W_EYE     * eye_signal
                     + W_PERCLOS * perclos_signal
                     + W_YAWN    * yaw_prob
                     + W_NOD     * nod_intensity)
            alert = score >= args.threshold

            # Terminal
            pitch_str = "calib…" if baseline_pitch is None else f"{pitch_offset:+5.1f}°"

            status = " NGANTUK  " if alert else " AWAKE     "
            print(
                f"\r  {status:<11} {score:>6.3f}  "
                f"{eye_label_disp:>8}({eye_conf_disp:.2f})  "
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
                    f"\nV2X EVENT [{ts}] ────────────────────────────────────\n"
                    f"fatigue_score : {score:.4f}\n"
                    f"eye_closure   : {eye_signal:.4f}    "
                    f"(label={eye_label_disp}, conf={eye_conf_disp:.2f})\n"
                    f"perclos       : {perclos*100:.1f}%   "
                    f"(threshold={PERCLOS_THRESHOLD*100:.0f}%)\n"
                    f"yawn_conf     : {yaw_label} @ {yaw_conf:.4f}\n"
                    f"pitch_offset  : {pitch_offset:+.2f}°  "
                    f"(baseline={baseline_pitch if baseline_pitch is not None else 0:+.2f}°)\n"
                    f"events/{ROLLING_WINDOW_SEC}s   : "
                    f"closures={closure_count}  yawns={yawn_count}  nods={nod_count}\n"
                    f"ACTION        : BROADCAST WARNING TO SURROUNDING UNITS"
                )

            if not args.no_display:
                draw_overlay(frame, (x1, y1, x2, y2),
                             eye_label_disp, eye_conf_disp,
                             yaw_label, yaw_conf,
                             score, yawn_count, nod_count, closure_count,
                             perclos, pitch_str, alert)
                cv2.putText(frame, f"FPS: {fps:.1f}", (fw - 100, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD, 1)
                cv2.imshow("REFLEX", frame)
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