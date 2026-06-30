"""
debug_mediapipe_viz.py
----------------------
Visualize MediaPipe Face Mesh landmarks with color-coded points:
  - All 468 landmarks   → small gray dots (context)
  - 6 EAR landmarks/eye → larger TEAL dots + labels (eye state detection)
  - 6 pose landmarks    → larger AMBER dots + labels (head nod via solvePnP)

Useful untuk:
  - Slide presentasi (screenshot frame yang clear)
  - Verifikasi sampling MediaPipe yang dipakai EAR & solvePnP
  - Penjelasan visual ke dosen

Usage:
  python debug_mediapipe_viz.py
  Press 's' to save current frame as mediapipe_debug_NNN.png
  Press 'q' to quit
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# Landmark sets (same as detect.py)
LEFT_EYE_EAR  = [33,  159, 158, 133, 153, 145]   # outer, up1, up2, inner, low1, low2
RIGHT_EYE_EAR = [263, 386, 385, 362, 380, 373]
POSE_IDS      = [1, 152, 33, 263, 61, 291]       # nose, chin, eye L outer, eye R outer, mouth L, mouth R
POSE_LABELS   = ["nose", "chin", "L-eye", "R-eye", "L-mouth", "R-mouth"]
EAR_LABELS    = ["p1", "p2", "p3", "p4", "p5", "p6"]

# Colors (BGR for OpenCV)
COLOR_ALL_DOTS = (180, 180, 180)   # light gray for context
COLOR_EYE      = (180, 160, 0)     # teal-ish (EAR landmarks)
COLOR_POSE     = (0, 165, 245)     # amber (pose landmarks)
COLOR_TEXT     = (255, 255, 255)


def draw_label(img, text, x, y, color):
    """Draw text with dark background pill for readability."""
    # (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    # cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 2), (30, 30, 30), -1)
    # cv2.putText(img, text, (x, y - 2),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)


def main():
    save_count = 0
    save_dir = Path("debug_frames")
    save_dir.mkdir(exist_ok=True)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[s] save frame   [q] quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # 1) Draw ALL 468 landmarks as small gray dots
            for pt in lm:
                x = int(pt.x * w)
                y = int(pt.y * h)
                cv2.circle(frame, (x, y), 1, COLOR_ALL_DOTS, -1)

            # 2) Draw 6 EAR landmarks per eye (larger teal)
            for idx, label_idx in zip(LEFT_EYE_EAR, EAR_LABELS):
                pt = lm[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                cv2.circle(frame, (x, y), 5, COLOR_EYE, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
                draw_label(frame, f"L-{label_idx}({idx})", x + 8, y, COLOR_EYE)
            for idx, label_idx in zip(RIGHT_EYE_EAR, EAR_LABELS):
                pt = lm[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                cv2.circle(frame, (x, y), 5, COLOR_EYE, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
                draw_label(frame, f"R-{label_idx}({idx})", x + 8, y, COLOR_EYE)

            # 3) Draw 6 pose landmarks (larger amber)
            for idx, label in zip(POSE_IDS, POSE_LABELS):
                pt = lm[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                cv2.circle(frame, (x, y), 6, COLOR_POSE, -1)
                cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
                draw_label(frame, f"{label}({idx})", x + 8, y + 18, COLOR_POSE)

        # Legend box
        legend_y = 30
        cv2.rectangle(frame, (10, 10), (320, 110), (30, 30, 30), -1)
        cv2.circle(frame, (25, legend_y), 5, COLOR_EYE, -1)
        cv2.putText(frame, "EAR landmarks (6 per eye)", (40, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.circle(frame, (25, legend_y + 30), 6, COLOR_POSE, -1)
        cv2.putText(frame, "Pose landmarks (solvePnP)", (40, legend_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.circle(frame, (25, legend_y + 60), 1, COLOR_ALL_DOTS, -1)
        cv2.putText(frame, "All 468 face mesh landmarks", (40, legend_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.imshow("MediaPipe Debug — [s] save  [q] quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_count += 1
            out_path = save_dir / f"mediapipe_debug_{save_count:03d}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"  saved: {out_path}")

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()