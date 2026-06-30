"""
debug_yolo_viz.py
-----------------
Visualize YOLO face detection + the mouth-crop region (lower 45% of bbox).
NO actual cropping — overlay only, untuk penjelasan visual rasio cropping.

Useful untuk:
  - Slide presentasi (yang menunjukkan dari mana yawn CNN ambil input)
  - Verifikasi bahwa mouth region selalu mencakup chin + mouth

Usage:
  python debug_yolo_viz.py
  Press 's' to save current frame as yolo_debug_NNN.png
  Press 'q' to quit
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MOUTH_CROP_START = 0.55   # same as detect.py — mouth starts at 55% from top of bbox
FACE_CONF_MIN    = 0.40
YOLO_WEIGHTS     = "models/yolo26n-face.pt"

# BGR
COLOR_FACE   = (245, 165, 0)     # teal-ish
COLOR_MOUTH  = (0, 165, 245)     # amber
COLOR_TEXT   = (255, 255, 255)


def draw_label(img, text, x, y, bg_color):
    """Label with colored background pill."""
    # (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    # cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), bg_color, -1)
    # cv2.putText(img, text, (x + 5, y - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)


def overlay_translucent(img, x1, y1, x2, y2, color, alpha=0.25):
    """Fill rectangle with translucent overlay."""
    sub = img[y1:y2, x1:x2].copy()
    rect = np.full(sub.shape, color, dtype=np.uint8)
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1 - alpha, rect, alpha, 0)


def main():
    save_count = 0
    save_dir = Path("debug_frames")
    save_dir.mkdir(exist_ok=True)

    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

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

        results = yolo(frame, verbose=False, conf=FACE_CONF_MIN, imgsz=640)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            # Pick largest face
            boxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[best])
            conf = float(confs[best])

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            face_h = y2 - y1
            face_w = x2 - x1

            # Mouth region (lower 45% of bbox)
            mouth_y1 = int(y1 + MOUTH_CROP_START * face_h)
            mouth_y2 = y2
            mouth_x1 = x1
            mouth_x2 = x2

            # 1) Face bbox (teal)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FACE, 2)
            draw_label(frame, f"Face  conf={conf:.2f}",
                       x1, y1 - 4, COLOR_FACE)

            # 2) Mouth region (amber translucent + border)
            overlay_translucent(frame, mouth_x1, mouth_y1, mouth_x2, mouth_y2,
                                COLOR_MOUTH, alpha=0.28)
            cv2.rectangle(frame, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2),
                          COLOR_MOUTH, 2)
            draw_label(frame, f"Mouth crop  (lower {int((1-MOUTH_CROP_START)*100)}% of bbox)",
                       mouth_x1, mouth_y2 + 22, COLOR_MOUTH)

            # 3) Dashed reference line at 55% from top (where mouth crop starts)
            for x_dash in range(x1, x2, 12):
                cv2.line(frame, (x_dash, mouth_y1), (x_dash + 6, mouth_y1),
                         COLOR_MOUTH, 1)

        #     # 4) Side annotations: heights
        #     cv2.putText(frame, f"h_face = {face_h}px",
        #                 (x2 + 8, y1 + 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE, 1, cv2.LINE_AA)
        #     cv2.putText(frame, f"h_mouth = {mouth_y2 - mouth_y1}px",
        #                 (x2 + 8, mouth_y1 + 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MOUTH, 1, cv2.LINE_AA)
        #     cv2.putText(frame, f"= 0.45 × h_face",
        #                 (x2 + 8, mouth_y1 + 38),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MOUTH, 1, cv2.LINE_AA)

        # Legend
        cv2.rectangle(frame, (10, 10), (310, 90), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, 25), (35, 40), COLOR_FACE, -1)
        cv2.putText(frame, "YOLO face bbox", (45, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 55), (35, 70), COLOR_MOUTH, -1)
        cv2.putText(frame, "Mouth crop region (input ke Yawn CNN)",
                    (45, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.imshow("YOLO Debug — [s] save  [q] quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_count += 1
            out_path = save_dir / f"yolo_debug_{save_count:03d}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"  saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()