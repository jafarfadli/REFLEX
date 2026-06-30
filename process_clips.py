import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp

BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
VIDEOS_DIR  = BASE_DIR / "validation_videos"
RESULTS_DIR = VIDEOS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# MUST MATCH detect.py
ALERT_THRESHOLD     = 0.65
W_EYE, W_PERCLOS, W_YAWN, W_NOD = 0.35, 0.15, 0.20, 0.30
EYE_MIN_DURATION    = 0.4
PERCLOS_WINDOW_SEC  = 60
PERCLOS_MIN_WINDOW_SEC  = 15
PERCLOS_THRESHOLD   = 0.15
YAWN_CONF_THRESHOLD = 0.70
YAWN_MIN_DURATION   = 1.0
NOD_THRESHOLD_DEG   = 12.0
NOD_MIN_DURATION    = 0.8
FACE_CONF_MIN       = 0.40
MOUTH_CROP_START    = 0.55
IMG_SIZE            = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEFT_EYE_EAR  = [33,  159, 158, 133, 153, 145]
RIGHT_EYE_EAR = [263, 386, 385, 362, 380, 373]
FACE_3D_MODEL = np.array([
    [   0.0,   0.0,   0.0], [   0.0, -63.6, -12.5],
    [ -43.3,  32.7, -26.0], [  43.3,  32.7, -26.0],
    [ -28.9, -28.9, -24.1], [  28.9, -28.9, -24.1],
], dtype=np.float64)
MP_POSE_IDS = [1, 152, 33, 263, 61, 291]


def _single_eye_ear(lm, ids, fw, fh):
    pts = np.array([[lm[i].x * fw, lm[i].y * fh] for i in ids], dtype=np.float64)
    vert  = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return float(vert / (2.0 * horiz)) if horiz > 1e-6 else 0.0


def compute_ear(lm, fw, fh):
    return (_single_eye_ear(lm, LEFT_EYE_EAR, fw, fh)
            + _single_eye_ear(lm, RIGHT_EYE_EAR, fw, fh)) / 2.0


def estimate_pitch(lm, fw, fh):
    image_pts = np.array([[lm[i].x * fw, lm[i].y * fh] for i in MP_POSE_IDS],
                          dtype=np.float64)
    focal = float(fw)
    cam = np.array([[focal, 0, fw/2], [0, focal, fh/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, image_pts, cam, np.zeros((4, 1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2))
    pitch = (np.arctan2(-rmat[1, 2], rmat[1, 1]) if sy < 1e-6
             else np.arctan2(rmat[2, 1], rmat[2, 2]))
    pd = float(np.degrees(pitch))
    return pd - 180 if pd > 90 else (pd + 180 if pd < -90 else pd)


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
        nn.Dropout(0.3), nn.Linear(model.last_channel, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(classes)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    return model, classes


@torch.no_grad()
def classify(model, classes, crop, transform):
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "unknown", 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], probs[idx].item()


def positive_prob(label, conf, positive_class):
    return conf if label == positive_class else 1.0 - conf


def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def mouth_crop(face_bgr):
    if face_bgr is None or face_bgr.size == 0: return None
    return face_bgr[int(face_bgr.shape[0] * MOUTH_CROP_START):, :]


def process_video(video_path, metadata, yolo, yaw_model, yaw_cls, transform, face_mesh):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = max(1.0, float(metadata.get("fps", 30.0)))
    cal = metadata["calibration"]
    ear_threshold     = cal["ear_threshold"]
    baseline_ear_open = cal["baseline_ear_open"]
    baseline_pitch    = cal["baseline_pitch"]

    # State (matches detect.py)
    closure_streak_start = None; closure_already_counted = False; closure_events = 0
    yawn_streak_start    = None; yawn_already_counted    = False; yawn_events    = 0
    nod_streak_start     = None; nod_already_counted     = False; nod_events     = 0
    perclos_history = deque()

    scores = []
    frame_idx = 0
    n_face = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        now = frame_idx / fps
        frame_idx += 1

        while perclos_history and now - perclos_history[0][0] > PERCLOS_WINDOW_SEC:
            perclos_history.popleft()

        fh, fw = frame.shape[:2]
        results = yolo(frame, verbose=False, conf=FACE_CONF_MIN, imgsz=640)
        detections = results[0].boxes

        ear_val = None
        current_pitch = None
        yaw_label, yaw_conf = "unknown", 0.0

        if detections is not None and len(detections) > 0:
            n_face += 1
            boxes = detections.xyxy.cpu().numpy()
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[best])
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)
            face_bgr = frame[y1:y2, x1:x2]
            m_crop = mouth_crop(face_bgr)

            mp_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if mp_results.multi_face_landmarks:
                lm = mp_results.multi_face_landmarks[0].landmark
                ear_val       = compute_ear(lm, fw, fh)
                current_pitch = estimate_pitch(lm, fw, fh)

            yaw_label, yaw_conf = classify(yaw_model, yaw_cls, m_crop, transform)

        is_eyes_closed = (ear_val is not None and ear_val < ear_threshold)

        if is_eyes_closed:
            if closure_streak_start is None:
                closure_streak_start = now; closure_already_counted = False
            if not closure_already_counted and now - closure_streak_start >= EYE_MIN_DURATION:
                closure_events += 1; closure_already_counted = True
        else:
            closure_streak_start = None; closure_already_counted = False

        if ear_val is not None:
            perclos_history.append((now, is_eyes_closed))

        # Only compute PERCLOS once window has enough history (avoids early saturation)
        window_age = now - perclos_history[0][0] if perclos_history else 0.0
        if window_age >= PERCLOS_MIN_WINDOW_SEC:
            perclos = sum(1 for _, c in perclos_history if c) / len(perclos_history)
        else:
            perclos = 0.0

        if ear_val is None:
            eye_signal = 0.0
        elif baseline_ear_open > ear_threshold:
            eye_signal = float(np.clip(
                (baseline_ear_open - ear_val) / (baseline_ear_open - ear_threshold), 0.0, 1.0,
            ))
        else:
            eye_signal = 1.0 if is_eyes_closed else 0.0

        yaw_prob = positive_prob(yaw_label, yaw_conf, "yawn")
        is_mouth_open = yaw_label == "yawn" and yaw_conf >= YAWN_CONF_THRESHOLD
        if is_mouth_open:
            if yawn_streak_start is None:
                yawn_streak_start = now; yawn_already_counted = False
            if not yawn_already_counted and now - yawn_streak_start >= YAWN_MIN_DURATION:
                yawn_events += 1; yawn_already_counted = True
        else:
            yawn_streak_start = None; yawn_already_counted = False

        nod_intensity = 0.0
        is_head_down = False
        if current_pitch is not None:
            pitch_offset  = current_pitch - baseline_pitch
            nod_intensity = min(abs(pitch_offset) / NOD_THRESHOLD_DEG, 1.0)
            is_head_down  = abs(pitch_offset) >= NOD_THRESHOLD_DEG

        if is_head_down:
            if nod_streak_start is None:
                nod_streak_start = now; nod_already_counted = False
            if not nod_already_counted and now - nod_streak_start >= NOD_MIN_DURATION:
                nod_events += 1; nod_already_counted = True
        else:
            nod_streak_start = None; nod_already_counted = False

        perclos_signal = min(perclos / PERCLOS_THRESHOLD, 1.0)
        score = (W_EYE * eye_signal + W_PERCLOS * perclos_signal
                 + W_YAWN * yaw_prob + W_NOD * nod_intensity)
        scores.append(score)

    cap.release()
    if not scores:
        return None

    max_score = float(max(scores))
    mean_score = float(np.mean(scores))
    predicted_label = "fatigue" if max_score >= ALERT_THRESHOLD else "non-fatigue"

    return {
        "id": metadata["id"], "video": video_path.name,
        "gt_label": metadata["label"], "predicted_label": predicted_label,
        "correct": predicted_label == metadata["label"],
        "max_score": max_score, "mean_score": mean_score,
        "yawn_count": yawn_events, "microsleep_count": closure_events, "nod_count": nod_events,
        "frames_total": frame_idx, "frames_face": n_face,
        "face_rate": n_face / max(frame_idx, 1),
        "ear_threshold": cal["ear_threshold"],
    }


def main():
    print(f"\n{'='*60}\n  Process Validation Clips\n{'='*60}")

    print("  Loading models...")
    yaw_path = MODELS_DIR / "yawn_clf.pt"
    if not yaw_path.exists():
        print(f"  [ERROR] {yaw_path} not found."); sys.exit(1)
    yaw_model, yaw_cls = load_cnn(yaw_path)

    from ultralytics import YOLO
    yolo_path = BASE_DIR / "models" / "yolo26n-face.pt"
    if not yolo_path.exists():
        print(f"  [ERROR] {yolo_path} not found."); sys.exit(1)
    yolo = YOLO(str(yolo_path))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    transform = get_transform()

    json_files = sorted(VIDEOS_DIR.glob("*.json"))
    if not json_files:
        print(f"  No clips in {VIDEOS_DIR}. Run record_clips.py first."); sys.exit(1)

    print(f"  Found {len(json_files)} clips. Processing...\n")
    print(f"  {'ID':>3}  {'GT':>11}  {'Pred':>11}  {'OK':>3}  "
          f"{'MaxS':>6}  {'MeanS':>6}  {'Yawn':>4}  {'Micro':>5}  {'Nod':>4}  {'Face%':>6}")
    print("  " + "-" * 82)

    results = []
    for jpath in json_files:
        try:
            meta = json.loads(jpath.read_text())
        except Exception as e:
            print(f"  [skip] {jpath.name}: {e}"); continue
        video_path = jpath.with_suffix(".mp4")
        if not video_path.exists():
            print(f"  [skip] {jpath.name}: no video"); continue
        r = process_video(video_path, meta, yolo, yaw_model, yaw_cls, transform, face_mesh)
        if r is None:
            print(f"  [skip] {jpath.name}: no frames"); continue
        results.append(r)
        ok_mark = "OK" if r["correct"] else "X"
        print(f"  {r['id']:>3}  {r['gt_label']:>11}  {r['predicted_label']:>11}  {ok_mark:>3}  "
              f"{r['max_score']:>6.3f}  {r['mean_score']:>6.3f}  {r['yawn_count']:>4d}  "
              f"{r['microsleep_count']:>5d}  {r['nod_count']:>4d}  {r['face_rate']*100:>5.1f}%")

    if not results:
        print("\n  No results."); return

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n
    tp = sum(1 for r in results if r["gt_label"] == "fatigue"     and r["predicted_label"] == "fatigue")
    fn = sum(1 for r in results if r["gt_label"] == "fatigue"     and r["predicted_label"] == "non-fatigue")
    fp = sum(1 for r in results if r["gt_label"] == "non-fatigue" and r["predicted_label"] == "fatigue")
    tn = sum(1 for r in results if r["gt_label"] == "non-fatigue" and r["predicted_label"] == "non-fatigue")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}\n  RESULTS  ({n} clips)\n{'='*60}")
    print(f"  Accuracy   : {accuracy*100:.1f}%  ({correct}/{n})")
    print(f"  Precision  : {precision:.3f}")
    print(f"  Recall     : {recall:.3f}")
    print(f"  F1 score   : {f1:.3f}")
    print(f"\n  Confusion matrix:")
    print(f"                   pred=fatigue  pred=non-fatigue")
    print(f"    gt=fatigue       {tp:>5}         {fn:>5}")
    print(f"    gt=non-fatigue   {fp:>5}         {tn:>5}")

    csv_path = RESULTS_DIR / "per_clip.csv"
    cols = ["id", "video", "gt_label", "predicted_label", "correct",
            "max_score", "mean_score", "yawn_count", "microsleep_count", "nod_count",
            "frames_total", "frames_face", "face_rate", "ear_threshold"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    md = [
        "# Validation Results — Per-Clip Analysis",
        "",
        f"- Total clips: **{n}**",
        f"- Accuracy: **{accuracy*100:.1f}%** ({correct}/{n})",
        f"- Precision: **{precision:.3f}**  ·  Recall: **{recall:.3f}**  ·  F1: **{f1:.3f}**",
        "",
        "## Confusion Matrix",
        "",
        "|                  | pred=fatigue | pred=non-fatigue |",
        "|------------------|:------------:|:----------------:|",
        f"| **gt=fatigue**     | {tp} | {fn} |",
        f"| **gt=non-fatigue** | {fp} | {tn} |",
        "",
        "## Per-Clip Details",
        "",
        "| ID | GT | Predicted | OK | MaxScore | MeanScore | Yawn | Microsleep | Nod | Face% |",
        "|----|------|------|:---:|--------:|---------:|----:|----:|----:|------:|",
    ]
    for r in results:
        ok = "OK" if r["correct"] else "X"
        md.append(
            f"| {r['id']} | {r['gt_label']} | {r['predicted_label']} | {ok} | "
            f"{r['max_score']:.3f} | {r['mean_score']:.3f} | "
            f"{r['yawn_count']} | {r['microsleep_count']} | "
            f"{r['nod_count']} | {r['face_rate']*100:.1f}% |"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(md))

    print(f"\n  CSV → {csv_path.relative_to(BASE_DIR)}")
    print(f"  MD  → {(RESULTS_DIR / 'summary.md').relative_to(BASE_DIR)}\n")


if __name__ == "__main__":
    main()