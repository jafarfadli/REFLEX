"""
record_validation.py
--------------------
Structured recording of labeled validation data for threshold/weight validation.

Records 8 predefined scenarios with ground-truth labels (FATIGUE / NORMAL).
For each frame, computes ALL detection signals and saves them to CSV.
The CSV is the input to evaluate.py for threshold sweep + ablation study.

Pipeline mirrors detect.py exactly (same calibration, same signal extraction)
so that validation results are directly applicable to the production pipeline.

Total session ~6 minutes:
  - Calibration         :  ~15 s
  - 8 scenarios × 20-30s:  ~3.5 min recording
  - Transitions         :  ~40 s

Usage:
  python record_validation.py
  python record_validation.py --camera 1
  python record_validation.py --output my_validation.csv
"""

import argparse
import csv
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
BASE_DIR     = Path(__file__).parent
MODELS_DIR   = BASE_DIR / "models"
OUTPUT_DIR   = BASE_DIR / "validation_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Detection config (MUST MATCH detect.py exactly) ──────────────────────────
W_EYE                 = 0.35
W_PERCLOS             = 0.15
W_YAWN                = 0.20
W_NOD                 = 0.30
ALERT_THRESHOLD       = 0.65

EYE_MIN_DURATION      = 0.4
PERCLOS_WINDOW_SEC    = 60
PERCLOS_THRESHOLD     = 0.15

YAWN_CONF_THRESHOLD   = 0.70
YAWN_MIN_DURATION     = 1.0

NOD_THRESHOLD_DEG     = 12.0
NOD_MIN_DURATION      = 0.8

CALIB_OPEN_SEC         = 5.0
CALIB_CLOSED_SEC       = 5.0
CALIB_COUNTDOWN_SEC    = 3
CALIB_MIN_SAMPLES      = 20

ROLLING_WINDOW_SEC    = 30
FACE_CONF_MIN         = 0.40
MOUTH_CROP_START      = 0.55
IMG_SIZE              = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_OPEN     = (0, 255, 0)
COLOR_CLOSED   = (0, 100, 255)
COLOR_NORMAL   = (0, 200, 80)
COLOR_FATIGUE  = (0, 100, 255)
WINDOW_NAME    = "Validation Recording"

# ── Landmarks ─────────────────────────────────────────────────────────────────
FACE_3D_MODEL = np.array([
    [   0.0,   0.0,   0.0],
    [   0.0, -63.6, -12.5],
    [ -43.3,  32.7, -26.0],
    [  43.3,  32.7, -26.0],
    [ -28.9, -28.9, -24.1],
    [  28.9, -28.9, -24.1],
], dtype=np.float64)
MP_POSE_IDS   = [1, 152, 33, 263, 61, 291]
LEFT_EYE_EAR  = [33,  159, 158, 133, 153, 145]
RIGHT_EYE_EAR = [263, 386, 385, 362, 380, 373]


# ── Scenarios ─────────────────────────────────────────────────────────────────
# label: 1 = FATIGUE, 0 = NORMAL
SCENARIOS = [
    {
        "name": "baseline_alert",
        "label": 0,
        "duration": 25,
        "instruction": "Look at camera normally, eyes open",
        "tip": "Blink naturally. Mild head movement OK.",
    },
    {
        "name": "talking_active",
        "label": 0,
        "duration": 20,
        "instruction": "Talk or read aloud — fully alert",
        "tip": "Mouth moves but NOT a yawn. Stay focused.",
    },
    {
        "name": "looking_around",
        "label": 0,
        "duration": 20,
        "instruction": "Check mirrors: look left, right, up, down",
        "tip": "Simulate active driving scan.",
    },
    {
        "name": "yawning",
        "label": 1,
        "duration": 30,
        "instruction": "Yawn deeply 3-4 times",
        "tip": "Each yawn ≥ 1 second wide mouth open.",
    },
    {
        "name": "eye_closure",
        "label": 1,
        "duration": 30,
        "instruction": "Close eyes 1-2s repeatedly (drowsy dozing)",
        "tip": "4-5 closures during this segment.",
    },
    {
        "name": "head_nodding",
        "label": 1,
        "duration": 30,
        "instruction": "Slowly nod head down (chin to chest), hold ≥1s",
        "tip": "SLOW sleepy nod, NOT a quick 'yes' nod. Repeat 3-4×.",
    },
    {
        "name": "drowsy_combo",
        "label": 1,
        "duration": 30,
        "instruction": "Combine: yawn + slow blinks + occasional head drop",
        "tip": "Most realistic drowsy-driver behavior.",
    },
    {
        "name": "recovery_alert",
        "label": 0,
        "duration": 20,
        "instruction": "Back to fully alert — wake up!",
        "tip": "Eyes open, head straight, focused.",
    },
]


# ── Helpers (identical to detect.py) ──────────────────────────────────────────

def _single_eye_ear(landmarks, ids, fw, fh) -> float:
    pts = np.array(
        [[landmarks[i].x * fw, landmarks[i].y * fh] for i in ids],
        dtype=np.float64,
    )
    vert  = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return float(vert / (2.0 * horiz)) if horiz > 1e-6 else 0.0


def compute_ear(landmarks, fw, fh):
    l = _single_eye_ear(landmarks, LEFT_EYE_EAR,  fw, fh)
    r = _single_eye_ear(landmarks, RIGHT_EYE_EAR, fw, fh)
    return (l + r) / 2.0


def estimate_pitch(landmarks, fw, fh):
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
    ok, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, image_pts, cam_matrix, dist_coeffs,
                                flags=cv2.SOLVEPNP_ITERATIVE)
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


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_cnn(model_path):
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
def classify(model, classes, crop, transform):
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "unknown", 0.0
    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    idx    = probs.argmax().item()
    return classes[idx], probs[idx].item()


def positive_prob(label, conf, positive_class):
    return conf if label == positive_class else 1.0 - conf


def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def mouth_crop(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return None
    fh = face_bgr.shape[0]
    return face_bgr[int(fh * MOUTH_CROP_START):, :]


# ── Calibration (identical to detect.py) ──────────────────────────────────────

def _show_countdown(cap, prompt, color, seconds):
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


def _capture_calib_phase(cap, face_mesh, prompt, color, duration, want_pitch):
    ear_values, pitch_values = [], []
    start_t = time.time()
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

        elapsed   = time.time() - start_t
        remaining = duration - elapsed
        cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"RECORDING: {prompt}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        ear_str = f"EAR = {ear_now:.4f}" if ear_now is not None else "no face"
        cv2.putText(frame, f"{remaining:4.1f}s left | {ear_str} | samples = {len(ear_values)}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        bar_w = int(w * (elapsed / duration))
        cv2.rectangle(frame, (0, 130), (bar_w, 136), color, -1)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    return ear_values, pitch_values


def run_calibration(cap, face_mesh):
    print("\n  ── CALIBRATION ────────────────────────────────────────")
    print("     Same as detect.py. 2 phases.\n")

    if not _show_countdown(cap, "look at camera with eyes OPEN",
                            COLOR_OPEN, CALIB_COUNTDOWN_SEC):
        return None
    print("     Phase 1/2: capturing OPEN baseline...")
    open_ears, pitch_vals = _capture_calib_phase(
        cap, face_mesh, "OPEN your eyes normally", COLOR_OPEN,
        CALIB_OPEN_SEC, want_pitch=True,
    )

    if not _show_countdown(cap, "CLOSE your eyes fully",
                            COLOR_CLOSED, CALIB_COUNTDOWN_SEC):
        return None
    print("     Phase 2/2: capturing CLOSED baseline...")
    closed_ears, _ = _capture_calib_phase(
        cap, face_mesh, "CLOSE your eyes fully", COLOR_CLOSED,
        CALIB_CLOSED_SEC, want_pitch=False,
    )

    if len(open_ears) < CALIB_MIN_SAMPLES or len(closed_ears) < CALIB_MIN_SAMPLES:
        print(f"     [ERROR] Not enough samples.")
        return None

    open_med   = float(np.median(open_ears))
    closed_med = float(np.median(closed_ears))
    threshold  = (open_med + closed_med) / 2.0
    pitch_med  = float(np.median(pitch_vals)) if pitch_vals else 0.0

    print(f"\n     EAR open / closed / threshold : "
          f"{open_med:.4f} / {closed_med:.4f} / {threshold:.4f}")
    print(f"     Pitch baseline                : {pitch_med:+.2f}°")
    return threshold, open_med, pitch_med


# ── Scenario recording ────────────────────────────────────────────────────────

def show_scenario_intro(cap, scenario, seconds=5):
    """Show scenario info with countdown before recording starts."""
    label_str = "FATIGUE" if scenario["label"] == 1 else "NORMAL"
    label_color = COLOR_FATIGUE if scenario["label"] == 1 else COLOR_NORMAL

    for sec in range(seconds, 0, -1):
        end_t = time.time() + 1.0
        while time.time() < end_t:
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            # Dark backdrop on top half
            cv2.rectangle(frame, (0, 0), (w, 240), (20, 20, 20), -1)
            cv2.putText(frame, f"NEXT: {scenario['name']}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Label: {label_str}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)
            cv2.putText(frame, scenario["instruction"], (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Tip: {scenario['tip']}", (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Duration: {scenario['duration']}s   |   Starting in {sec}...",
                        (20, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                return False
    return True


def record_scenario(cap, face_mesh, yolo, yaw_model, yaw_cls, transform,
                    ear_threshold, baseline_ear_open, baseline_pitch,
                    scenario, state, csv_writer, frame_idx_counter):
    """Record one scenario, writing per-frame data to CSV. Returns updated frame_idx_counter."""

    start_t  = time.time()
    duration = scenario["duration"]
    name     = scenario["name"]
    gt_label = scenario["label"]
    label_color = COLOR_FATIGUE if gt_label == 1 else COLOR_NORMAL

    print(f"\n  Recording '{name}' (label={gt_label}, duration={duration}s)...")

    frames_in_scenario = 0

    while True:
        elapsed = time.time() - start_t
        if elapsed >= duration:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        fh, fw = frame.shape[:2]

        # Prune rolling windows
        while state["yawn_times"]      and now - state["yawn_times"][0]         > ROLLING_WINDOW_SEC:
            state["yawn_times"].popleft()
        while state["nod_times"]       and now - state["nod_times"][0]          > ROLLING_WINDOW_SEC:
            state["nod_times"].popleft()
        while state["closure_events"]  and now - state["closure_events"][0]     > ROLLING_WINDOW_SEC:
            state["closure_events"].popleft()
        while state["perclos_history"] and now - state["perclos_history"][0][0] > PERCLOS_WINDOW_SEC:
            state["perclos_history"].popleft()

        # YOLO
        results    = yolo(frame, verbose=False, conf=FACE_CONF_MIN, imgsz=640)
        detections = results[0].boxes

        # Initialize per-frame values
        ear_val          = None
        current_pitch    = None
        yaw_label        = "unknown"
        yaw_conf         = 0.0
        is_eyes_closed   = False
        is_mouth_open    = False
        is_head_down     = False
        eye_signal       = 0.0
        yaw_prob         = 0.0
        nod_intensity    = 0.0
        pitch_offset     = 0.0
        perclos          = 0.0
        face_detected    = False

        if detections is not None and len(detections) > 0:
            face_detected = True
            boxes = detections.xyxy.cpu().numpy()
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best  = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[best])
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)
            face_bgr = frame[y1:y2, x1:x2]
            m_crop   = mouth_crop(face_bgr)

            # MediaPipe
            mp_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if mp_results.multi_face_landmarks:
                lm = mp_results.multi_face_landmarks[0].landmark
                ear_val       = compute_ear(lm, fw, fh)
                current_pitch = estimate_pitch(lm, fw, fh)

            # Eye state
            is_eyes_closed = (ear_val is not None and ear_val < ear_threshold)

            # Sustained closure
            if is_eyes_closed:
                if state["closure_streak_start"] is None:
                    state["closure_streak_start"]    = now
                    state["closure_already_counted"] = False
                if (not state["closure_already_counted"]
                        and now - state["closure_streak_start"] >= EYE_MIN_DURATION):
                    state["closure_events"].append(now)
                    state["closure_already_counted"] = True
            else:
                state["closure_streak_start"]    = None
                state["closure_already_counted"] = False

            # PERCLOS
            if ear_val is not None:
                state["perclos_history"].append((now, is_eyes_closed))
            if state["perclos_history"]:
                perclos = sum(1 for _, c in state["perclos_history"] if c) / len(state["perclos_history"])

            # Eye signal for fusion
            if ear_val is None:
                eye_signal = 0.0
            elif baseline_ear_open > ear_threshold:
                eye_signal = float(np.clip(
                    (baseline_ear_open - ear_val) / (baseline_ear_open - ear_threshold),
                    0.0, 1.0,
                ))
            else:
                eye_signal = 1.0 if is_eyes_closed else 0.0

            # Yawn
            yaw_label, yaw_conf = classify(yaw_model, yaw_cls, m_crop, transform)
            yaw_prob = positive_prob(yaw_label, yaw_conf, "yawn")
            is_mouth_open = yaw_label == "yawn" and yaw_conf >= YAWN_CONF_THRESHOLD
            if is_mouth_open:
                if state["yawn_streak_start"] is None:
                    state["yawn_streak_start"]    = now
                    state["yawn_already_counted"] = False
                if (not state["yawn_already_counted"]
                        and now - state["yawn_streak_start"] >= YAWN_MIN_DURATION):
                    state["yawn_times"].append(now)
                    state["yawn_already_counted"] = True
            else:
                state["yawn_streak_start"]    = None
                state["yawn_already_counted"] = False

            # Nod
            if current_pitch is not None:
                pitch_offset  = current_pitch - baseline_pitch
                nod_intensity = min(abs(pitch_offset) / NOD_THRESHOLD_DEG, 1.0)
                is_head_down  = abs(pitch_offset) >= NOD_THRESHOLD_DEG
            if is_head_down:
                if state["nod_streak_start"] is None:
                    state["nod_streak_start"]    = now
                    state["nod_already_counted"] = False
                if (not state["nod_already_counted"]
                        and now - state["nod_streak_start"] >= NOD_MIN_DURATION):
                    state["nod_times"].append(now)
                    state["nod_already_counted"] = True
            else:
                state["nod_streak_start"]    = None
                state["nod_already_counted"] = False

        # Fusion (with default weights)
        perclos_signal = min(perclos / PERCLOS_THRESHOLD, 1.0)
        score_default  = (W_EYE     * eye_signal
                          + W_PERCLOS * perclos_signal
                          + W_YAWN    * yaw_prob
                          + W_NOD     * nod_intensity)
        alert_default  = score_default >= ALERT_THRESHOLD

        # Write CSV row
        csv_writer.writerow({
            "frame_idx":       frame_idx_counter[0],
            "timestamp":       round(now, 4),
            "scenario":        name,
            "gt_label":        gt_label,
            "face_detected":   int(face_detected),
            "ear_value":       round(ear_val, 6) if ear_val is not None else "",
            "is_eyes_closed":  int(is_eyes_closed),
            "eye_signal":      round(eye_signal, 6),
            "yawn_label":      yaw_label,
            "yawn_conf":       round(yaw_conf, 6),
            "yawn_prob":       round(yaw_prob, 6),
            "is_mouth_open":   int(is_mouth_open),
            "pitch_value":     round(current_pitch, 4) if current_pitch is not None else "",
            "pitch_offset":    round(pitch_offset, 4),
            "nod_intensity":   round(nod_intensity, 6),
            "is_head_down":    int(is_head_down),
            "perclos":         round(perclos, 6),
            "perclos_signal":  round(perclos_signal, 6),
            "closure_count":   len(state["closure_events"]),
            "yawn_count":      len(state["yawn_times"]),
            "nod_count":       len(state["nod_times"]),
            "score_default":   round(score_default, 6),
            "alert_default":   int(alert_default),
        })
        frame_idx_counter[0] += 1
        frames_in_scenario  += 1

        # Live overlay
        remaining = duration - elapsed
        progress  = elapsed / duration
        bar_w     = int(fw * progress)

        cv2.rectangle(frame, (0, 0), (fw, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"REC [{name}]  label={'FATIGUE' if gt_label == 1 else 'NORMAL'}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color, 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"{remaining:4.1f}s | EAR {ear_val:.3f}" if ear_val else f"{remaining:4.1f}s | no face",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 90), (bar_w, 96), label_color, -1)

        # Score indicator
        score_color = (0, 0, 220) if alert_default else (40, 200, 80)
        cv2.putText(frame, f"score = {score_default:.3f}",
                    (fw - 180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n  Aborted by user.")
            return frame_idx_counter[0], False

    print(f"     ✓ saved {frames_in_scenario} frames for '{name}'")
    return frame_idx_counter[0], True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structured recording for validation")
    parser.add_argument("--camera",  type=int,    default=0)
    parser.add_argument("--output",  type=str,    default="validation.csv")
    args = parser.parse_args()

    output_path = OUTPUT_DIR / args.output

    print(f"\n{'='*60}")
    print(f"  Validation Recording  —  Structured Scenarios")
    print(f"{'='*60}")
    print(f"  Camera     : {args.camera}")
    print(f"  Output CSV : {output_path.relative_to(BASE_DIR)}")
    print(f"  Scenarios  : {len(SCENARIOS)}")
    total_rec = sum(s["duration"] for s in SCENARIOS)
    print(f"  Total rec  : {total_rec}s (~{total_rec/60:.1f} min)")

    # ── Load models ───────────────────────────────────────────────────────────
    yaw_path = MODELS_DIR / "yawn_clf.pt"
    if not yaw_path.exists():
        print(f"\n  [ERROR] yawn_clf model not found: {yaw_path}\n")
        sys.exit(1)

    print("\n  Loading yawn CNN...")
    yaw_model, yaw_cls = load_cnn(yaw_path)

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

    print("  Loading MediaPipe Face Mesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    transform = get_transform()

    # ── Webcam ────────────────────────────────────────────────────────────────
    print(f"\n  Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open camera index {args.camera}")
        sys.exit(1)

    # ── Calibration ───────────────────────────────────────────────────────────
    calib = run_calibration(cap, face_mesh)
    if calib is None:
        print("\n  Calibration failed. Exiting.\n")
        cap.release()
        sys.exit(1)
    ear_threshold, baseline_ear_open, baseline_pitch = calib

    # ── Session state (persists across scenarios) ─────────────────────────────
    state = {
        "closure_streak_start":    None,
        "closure_already_counted": False,
        "closure_events":          deque(),
        "perclos_history":         deque(),
        "yawn_streak_start":       None,
        "yawn_already_counted":    False,
        "yawn_times":              deque(),
        "nod_streak_start":        None,
        "nod_already_counted":     False,
        "nod_times":               deque(),
    }

    # ── Open CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "frame_idx", "timestamp", "scenario", "gt_label",
        "face_detected",
        "ear_value", "is_eyes_closed", "eye_signal",
        "yawn_label", "yawn_conf", "yawn_prob", "is_mouth_open",
        "pitch_value", "pitch_offset", "nod_intensity", "is_head_down",
        "perclos", "perclos_signal",
        "closure_count", "yawn_count", "nod_count",
        "score_default", "alert_default",
    ]
    csv_file   = open(output_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Metadata header (as a comment row)
    meta = {f: "" for f in fieldnames}
    meta["frame_idx"] = "# metadata"
    meta["ear_value"] = f"baseline_ear_open={baseline_ear_open:.4f}"
    meta["pitch_value"] = f"baseline_pitch={baseline_pitch:.4f}"
    meta["eye_signal"] = f"ear_threshold={ear_threshold:.4f}"
    csv_writer.writerow(meta)

    # ── Run scenarios ─────────────────────────────────────────────────────────
    frame_idx_counter = [0]
    try:
        for i, scenario in enumerate(SCENARIOS, start=1):
            print(f"\n  ── Scenario {i}/{len(SCENARIOS)}: {scenario['name']} ──")
            if not show_scenario_intro(cap, scenario, seconds=5):
                print("  Aborted.")
                break
            _, ok = record_scenario(
                cap, face_mesh, yolo, yaw_model, yaw_cls, transform,
                ear_threshold, baseline_ear_open, baseline_pitch,
                scenario, state, csv_writer, frame_idx_counter,
            )
            if not ok:
                break
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    finally:
        csv_file.close()
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_frames = frame_idx_counter[0]
    print(f"\n{'='*60}")
    print("  Recording complete")
    print(f"{'='*60}")
    print(f"  Total frames saved : {total_frames}")
    print(f"  CSV file           : {output_path}")
    print(f"\n  Calibration values (for reference):")
    print(f"    baseline_ear_open : {baseline_ear_open:.4f}")
    print(f"    ear_threshold     : {ear_threshold:.4f}")
    print(f"    baseline_pitch    : {baseline_pitch:+.4f}°")
    print(f"\n  Next step:")
    print(f"    python evaluate.py --csv {output_path.relative_to(BASE_DIR)}")
    print()


if __name__ == "__main__":
    main()