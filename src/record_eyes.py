import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
OUTPUT_BASE = BASE_DIR / "dataset" / "eye_dataset"

# ── Config ────────────────────────────────────────────────────────────────────
PHASE_DURATION_SEC = 60
SAVE_FPS_TARGET    = 10
COUNTDOWN_SEC      = 5
FACE_CONF_MIN      = 0.40
EYE_PAD_RATIO      = 0.4

# ── MediaPipe eye contour landmarks ───────────────────────────────────────────
LEFT_EYE_IDS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_IDS = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466,
]


# ── Eye crop function (MUST match detect.py) ─────────────────────────────────

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


# ── Recording phase ───────────────────────────────────────────────────────────

def countdown(cap, label: str, seconds: int):
    """Display a live preview with countdown text."""
    end = time.time() + seconds
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            continue
        remaining = int(end - time.time()) + 1
        h, w = frame.shape[:2]
        msg = f"Get ready: {label.upper()} in {remaining}..."
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, msg, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Eye Recorder", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            return False
    return True


def record_phase(cap, yolo, face_mesh, label: str,
                 duration_sec: int, save_fps: int) -> int:
    """Record one phase. Returns the number of crop images saved."""
    out_dir = OUTPUT_BASE / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve existing files; pick up from the next index
    existing = [p for p in out_dir.glob("*.jpg")]
    idx = len(existing)

    print(f"\n  Recording '{label}' for {duration_sec}s...")
    print("  Tip: vary expression + slight head movement → stronger model.")
    print("  Press 'q' to abort early.\n")

    save_interval = 1.0 / save_fps
    t_start       = time.time()
    last_save     = 0.0
    saved_now     = 0
    no_face_count = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration_sec:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        h_fr, w_fr = frame.shape[:2]

        # Face detection
        results    = yolo(frame, verbose=False, conf=FACE_CONF_MIN, imgsz=640)
        detections = results[0].boxes

        eye_crops = []
        if detections is not None and len(detections) > 0:
            mp_results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if mp_results.multi_face_landmarks:
                lm = mp_results.multi_face_landmarks[0].landmark
                eye_crops = extract_eye_crops(frame, lm, w_fr, h_fr)
                no_face_count = 0
            else:
                no_face_count += 1
        else:
            no_face_count += 1

        # Save at fixed cadence (avoids 1000s of nearly-identical frames)
        now = time.time()
        if eye_crops and (now - last_save) >= save_interval:
            for crop in eye_crops:
                fname = out_dir / f"{label}_{idx:05d}.jpg"
                cv2.imwrite(str(fname), crop)
                idx       += 1
                saved_now += 1
            last_save = now

        # Live preview overlay
        remaining = duration_sec - elapsed
        progress  = min(elapsed / duration_sec, 1.0)
        bar_w     = int(w_fr * progress)

        cv2.rectangle(frame, (0, 0), (w_fr, 70), (0, 0, 0), -1)
        color = (0, 255, 0) if label == "open" else (0, 165, 255)
        cv2.putText(frame, f"REC '{label.upper()}'  {remaining:4.1f}s left",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"saved: {saved_now}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        # Progress bar
        cv2.rectangle(frame, (0, 70), (bar_w, 76), color, -1)

        # Warn if MediaPipe isn't seeing the face
        if no_face_count > 15:
            cv2.putText(frame, "!! face not detected — adjust position !!",
                        (20, h_fr - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Show one of the most recent crops as a thumbnail (sanity check)
        if eye_crops:
            thumb = cv2.resize(eye_crops[0], (96, 96))
            frame[h_fr - 96 - 10 : h_fr - 10, w_fr - 96 - 10 : w_fr - 10] = thumb
            cv2.rectangle(frame,
                          (w_fr - 96 - 10, h_fr - 96 - 10),
                          (w_fr - 10, h_fr - 10),
                          (255, 255, 255), 1)

        cv2.imshow("Eye Recorder", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n  Aborted by user.")
            return saved_now

    return saved_now


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record eye dataset from webcam")
    parser.add_argument("--camera",   type=int, default=0,                 help="Webcam index")
    parser.add_argument("--duration", type=int, default=PHASE_DURATION_SEC, help="Seconds per phase")
    parser.add_argument("--phase",    choices=["open", "closed", "both"], default="both",
                        help="Which phase(s) to record")
    parser.add_argument("--fps",      type=int, default=SAVE_FPS_TARGET,
                        help="Target save FPS (images/sec)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Eye Dataset Recorder")
    print(f"{'='*60}")
    print(f"  Camera     : {args.camera}")
    print(f"  Duration   : {args.duration} s per phase")
    print(f"  Save FPS   : {args.fps}")
    print(f"  Output dir : {OUTPUT_BASE.relative_to(BASE_DIR)}/")

    # Load models
    print("\n  Loading YOLO face detector...")
    try:
        from ultralytics import YOLO
        yolo_path = BASE_DIR / "yolo26n-face.pt"
        if not yolo_path.exists():
            print(f"\n  [ERROR] YOLO weights not found: {yolo_path}\n")
            sys.exit(1)
        yolo = YOLO(str(yolo_path))
    except Exception as e:
        print(f"\n  [ERROR] Failed to load YOLO: {e}\n")
        sys.exit(1)

    print("  Loading MediaPipe Face Mesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open camera
    print(f"\n  Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open camera index {args.camera}\n")
        sys.exit(1)

    phases = ["open", "closed"] if args.phase == "both" else [args.phase]

    totals = {}
    try:
        for i, phase in enumerate(phases):
            label_msg = "open your eyes naturally" if phase == "open" \
                        else "fully close your eyes"
            print(f"\n  ── Phase {i+1}/{len(phases)}: {phase.upper()} "
                  f"({label_msg}) ──")
            if not countdown(cap, phase, COUNTDOWN_SEC):
                break
            saved = record_phase(cap, yolo, face_mesh, phase,
                                 duration_sec=args.duration, save_fps=args.fps)
            totals[phase] = saved
            print(f"  Phase complete. Saved {saved} crops.")

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    # Summary
    print(f"\n{'='*60}")
    print("  Recording Summary")
    print(f"{'='*60}")
    for phase in ("open", "closed"):
        out_dir = OUTPUT_BASE / phase
        if out_dir.exists():
            count = len(list(out_dir.glob("*.jpg")))
            saved_now = totals.get(phase, 0)
            print(f"  {phase:<7}  total in folder: {count:5d}   "
                  f"saved this run: {saved_now}")

if __name__ == "__main__":
    main()
