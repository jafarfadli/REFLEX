import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from src.face_mesh_compat import FaceMeshCompat

BASE_DIR    = Path(__file__).parent
VIDEOS_DIR  = BASE_DIR / "validation_videos"
CALIB_FILE  = BASE_DIR / "calibration.json"
VIDEOS_DIR.mkdir(exist_ok=True)

CLIP_DURATION     = 10.0
COUNTDOWN_SEC     = 3
CALIB_OPEN_SEC    = 5.0
CALIB_CLOSED_SEC  = 5.0
CALIB_MIN_SAMPLES = 20

SOUND_RECORD_START = "record_start"
SOUND_RECORD_END   = "record_end"
SOUND_PHASE_DONE   = "phase_done"
SOUND_CALIB_DONE   = "calib_done"

_MACOS_SOUNDS = {
    SOUND_RECORD_START: "/System/Library/Sounds/Pop.aiff",
    SOUND_RECORD_END:   "/System/Library/Sounds/Ping.aiff",
    SOUND_PHASE_DONE:   "/System/Library/Sounds/Glass.aiff",
    SOUND_CALIB_DONE:   "/System/Library/Sounds/Hero.aiff",
}
_LINUX_SOUNDS = {
    SOUND_RECORD_START: "/usr/share/sounds/freedesktop/stereo/dialog-information.oga",
    SOUND_RECORD_END:   "/usr/share/sounds/freedesktop/stereo/complete.oga",
    SOUND_PHASE_DONE:   "/usr/share/sounds/freedesktop/stereo/bell.oga",
    SOUND_CALIB_DONE:   "/usr/share/sounds/freedesktop/stereo/complete.oga",
}

IS_MACOS   = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

LEFT_EYE_EAR  = [33,  159, 158, 133, 153, 145]
RIGHT_EYE_EAR = [263, 386, 385, 362, 380, 373]
FACE_3D_MODEL = np.array([
    [   0.0,   0.0,   0.0], [   0.0, -63.6, -12.5],
    [ -43.3,  32.7, -26.0], [  43.3,  32.7, -26.0],
    [ -28.9, -28.9, -24.1], [  28.9, -28.9, -24.1],
], dtype=np.float64)
MP_POSE_IDS = [1, 152, 33, 263, 61, 291]

COLOR_OPEN   = (0, 255, 0)
COLOR_CLOSED = (0, 100, 255)
WINDOW_NAME  = "Record Validation Clips"


def play_sound(sound_type):
    try:
        if IS_MACOS:
            path = _MACOS_SOUNDS.get(sound_type)
            if path:
                subprocess.Popen(["afplay", path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif IS_WINDOWS:
            import winsound
            flag_map = {
                SOUND_RECORD_START: winsound.MB_OK,
                SOUND_RECORD_END:   winsound.MB_ICONASTERISK,
                SOUND_PHASE_DONE:   winsound.MB_ICONASTERISK,
                SOUND_CALIB_DONE:   winsound.MB_ICONEXCLAMATION,
            }
            winsound.MessageBeep(flag_map.get(sound_type, winsound.MB_OK))
        elif IS_LINUX:
            path = _LINUX_SOUNDS.get(sound_type)
            if path:
                try:
                    subprocess.Popen(["paplay", path],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    print("\a", end="", flush=True)
    except Exception:
        pass


def _single_eye_ear(lm, ids, fw, fh):
    pts = np.array([[lm[i].x * fw, lm[i].y * fh] for i in ids], dtype=np.float64)
    vert  = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return float(vert / (2.0 * horiz)) if horiz > 1e-6 else 0.0


def compute_ear(lm, fw, fh):
    return (_single_eye_ear(lm, LEFT_EYE_EAR, fw, fh)
            + _single_eye_ear(lm, RIGHT_EYE_EAR, fw, fh)) / 2.0


def estimate_pitch(lm, fw, fh):
    image_pts = np.array(
        [[lm[i].x * fw, lm[i].y * fh] for i in MP_POSE_IDS], dtype=np.float64,
    )
    focal = float(fw)
    cam = np.array([[focal, 0, fw/2], [0, focal, fh/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, image_pts, cam, np.zeros((4, 1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2))
    pitch = (np.arctan2(-rmat[1, 2], rmat[1, 1]) if sy < 1e-6
             else np.arctan2(rmat[2, 1], rmat[2, 2]))
    pd = float(np.degrees(pitch))
    return pd - 180 if pd > 90 else (pd + 180 if pd < -90 else pd)


def load_calibration():
    if not CALIB_FILE.exists():
        return None
    try:
        data = json.loads(CALIB_FILE.read_text())
        ear_t  = float(data["ear_threshold"])
        bo_ear = float(data["baseline_ear_open"])
        b_pit  = float(data["baseline_pitch"])
        if not (0.05 < ear_t < 0.50 and 0.10 < bo_ear < 0.60 and -45 < b_pit < 45):
            return None
        return {"ear_threshold": ear_t, "baseline_ear_open": bo_ear,
                "baseline_pitch": b_pit, "saved_at": data.get("saved_at", "unknown")}
    except Exception:
        return None


def save_calibration(ear_t, bo_ear, b_pit):
    payload = {
        "version": 1, "ear_threshold": float(ear_t),
        "baseline_ear_open": float(bo_ear), "baseline_pitch": float(b_pit),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.system(),
    }
    try:
        CALIB_FILE.write_text(json.dumps(payload, indent=2))
        return True
    except Exception:
        return False


def _show_countdown(cap, prompt, color, seconds):
    for sec in range(seconds, 0, -1):
        end_t = time.time() + 1.0
        while time.time() < end_t:
            ret, frame = cap.read()
            if not ret: continue
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


def _capture_phase(cap, face_mesh, prompt, color, duration, want_pitch):
    ear_values, pitch_values = [], []
    start_t = time.time()
    while time.time() - start_t < duration:
        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ear_now = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear_now = compute_ear(lm, w, h)
            ear_values.append(ear_now)
            if want_pitch:
                p = estimate_pitch(lm, w, h)
                if p is not None: pitch_values.append(p)
        elapsed = time.time() - start_t
        remaining = duration - elapsed
        cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"RECORDING: {prompt}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        ear_str = f"EAR = {ear_now:.4f}" if ear_now is not None else "no face"
        cv2.putText(frame, f"{remaining:4.1f}s left | {ear_str} | n = {len(ear_values)}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        bar_w = int(w * (elapsed / duration))
        cv2.rectangle(frame, (0, 130), (bar_w, 136), color, -1)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    return ear_values, pitch_values


def run_calibration(cap, face_mesh):
    print("\n  ── CALIBRATION ──")
    if not _show_countdown(cap, "eyes OPEN", COLOR_OPEN, COUNTDOWN_SEC):
        return None
    open_ears, pitch_vals = _capture_phase(cap, face_mesh, "OPEN your eyes", COLOR_OPEN,
                                            CALIB_OPEN_SEC, True)
    play_sound(SOUND_PHASE_DONE)
    if not _show_countdown(cap, "eyes CLOSED", COLOR_CLOSED, COUNTDOWN_SEC):
        return None
    closed_ears, _ = _capture_phase(cap, face_mesh, "CLOSE eyes fully", COLOR_CLOSED,
                                     CALIB_CLOSED_SEC, False)
    play_sound(SOUND_CALIB_DONE)
    if len(open_ears) < CALIB_MIN_SAMPLES or len(closed_ears) < CALIB_MIN_SAMPLES:
        return None
    open_med   = float(np.median(open_ears))
    closed_med = float(np.median(closed_ears))
    threshold  = (open_med * 1 + closed_med * 2) / 3
    pitch_med  = float(np.median(pitch_vals)) if pitch_vals else 0.0
    print(f"     EAR thr: {threshold:.4f}  |  pitch baseline: {pitch_med:+.2f}°")
    return threshold, open_med, pitch_med


def get_next_clip_id():
    existing = list(VIDEOS_DIR.glob("*.mp4"))
    ids = []
    for p in existing:
        try: ids.append(int(p.stem.split("_")[0]))
        except ValueError: pass
    return (max(ids) + 1) if ids else 1


def run_standby(cap, calibration):
    while True:
        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

        n_existing = len(list(VIDEOS_DIR.glob("*.mp4")))
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, "STANDBY", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Clips: {n_existing}  ·  Next ID: {get_next_clip_id():03d}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR thr: {calibration['ear_threshold']:.3f}",
                    (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pitch base: {calibration['baseline_pitch']:+.1f}°",
                    (w - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        y0 = h - 130
        cv2.rectangle(frame, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "[r]  Start recording (10s)", (20, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "[c]  Re-calibrate", (20, y0 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "[q]  Quit", (20, y0 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"): return "QUIT"
        if key == ord("r"): return "RECORD"
        if key == ord("c"): return "RECALIBRATE"


def record_clip(cap):
    for sec in range(COUNTDOWN_SEC, 0, -1):
        end_t = time.time() + 1.0
        while time.time() < end_t:
            ret, frame = cap.read()
            if not ret: continue
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
            cv2.putText(frame, "RECORDING IN...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{sec}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(30)

    play_sound(SOUND_RECORD_START)
    frames = []
    start_t = time.time()
    while time.time() - start_t < CLIP_DURATION:
        ret, frame = cap.read()
        if not ret: continue
        frames.append(frame.copy())   # raw frame WITHOUT overlay
        elapsed   = time.time() - start_t
        remaining = CLIP_DURATION - elapsed
        h, w = frame.shape[:2]
        display = frame.copy()
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 200), -1)
        cv2.putText(display, f"REC  {remaining:4.1f}s", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        bar_w = int(w * (elapsed / CLIP_DURATION))
        cv2.rectangle(display, (0, 80), (bar_w, 86), (0, 0, 200), -1)
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)
    play_sound(SOUND_RECORD_END)
    actual_fps = len(frames) / CLIP_DURATION
    return frames, actual_fps


def prompt_label(frames):
    preview = frames[len(frames) // 2].copy()
    h, w = preview.shape[:2]
    while True:
        display = preview.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.5, overlay, 0.5, 0)

        cv2.rectangle(display, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.putText(display, "LABEL THIS CLIP", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display,
                    f"{len(frames)} frames  ·  ~{len(frames)/CLIP_DURATION:.1f} fps",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        y0 = h - 150
        cv2.rectangle(display, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.putText(display, "[f]  FATIGUE", (20, y0 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "[n]  NON-FATIGUE", (20, y0 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 200, 80), 2, cv2.LINE_AA)
        cv2.putText(display, "[d]  Discard", (20, y0 + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("f"): return "fatigue"
        if key == ord("n"): return "non-fatigue"
        if key == ord("d"): return "discard"


def save_clip(frames, fps, label, calibration):
    clip_id  = get_next_clip_id()
    base     = f"{clip_id:03d}_{label}"
    video_p  = VIDEOS_DIR / f"{base}.mp4"
    meta_p   = VIDEOS_DIR / f"{base}.json"
    h, w     = frames[0].shape[:2]
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(video_p), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    metadata = {
        "id": clip_id, "label": label,
        "duration_sec": len(frames) / fps, "frame_count": len(frames),
        "fps": fps, "width": w, "height": h,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration": {
            "ear_threshold":     calibration["ear_threshold"],
            "baseline_ear_open": calibration["baseline_ear_open"],
            "baseline_pitch":    calibration["baseline_pitch"],
        },
    }
    meta_p.write_text(json.dumps(metadata, indent=2))
    return video_p

def make_face_mesh():
    """Auto-pick API based on mediapipe version."""
    import mediapipe as mp
    try:
        # Try legacy first (faster for older setups)
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
    except (AttributeError, ImportError):
        return FaceMeshCompat(
            model_path=BASE_DIR / "face_landmarker.task",
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )


def main():
    print(f"\n{'='*60}\n  Record Validation Clips\n{'='*60}")
    print(f"  Output: {VIDEOS_DIR}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera"); sys.exit(1)

    face_mesh = make_face_mesh()

    calibration = load_calibration()
    if calibration:
        print(f"  Loaded calibration (saved {calibration['saved_at']}):")
        print(f"    EAR threshold: {calibration['ear_threshold']:.4f}")
        print(f"    Pitch base   : {calibration['baseline_pitch']:+.2f}°")
    else:
        print(f"\n  No calibration found. Starting calibration...")
        result = run_calibration(cap, face_mesh)
        if result is None:
            print("  Calibration failed."); cap.release(); sys.exit(1)
        ear_t, bo, bp = result
        save_calibration(ear_t, bo, bp)
        calibration = {"ear_threshold": ear_t, "baseline_ear_open": bo, "baseline_pitch": bp}

    print(f"\n  Press 'r' to record, 'c' to re-calibrate, 'q' to quit.\n")

    try:
        while True:
            action = run_standby(cap, calibration)
            if action == "QUIT":
                break
            if action == "RECALIBRATE":
                result = run_calibration(cap, face_mesh)
                if result:
                    ear_t, bo, bp = result
                    save_calibration(ear_t, bo, bp)
                    calibration = {"ear_threshold": ear_t,
                                   "baseline_ear_open": bo, "baseline_pitch": bp}
                continue
            if action == "RECORD":
                print(f"\n  Recording clip {get_next_clip_id():03d}...")
                frames, fps = record_clip(cap)
                print(f"    Captured {len(frames)} frames at {fps:.1f} fps")
                label = prompt_label(frames)
                if label == "discard":
                    print("    Discarded."); continue
                video_path = save_clip(frames, fps, label, calibration)
                print(f"    Saved: {video_path.name}  [label={label}]")
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        cap.release(); face_mesh.close(); cv2.destroyAllWindows()
        n = len(list(VIDEOS_DIR.glob("*.mp4")))
        print(f"\n  Total clips: {n}")
        print(f"  Next: python process_clips.py\n")


if __name__ == "__main__":
    main()