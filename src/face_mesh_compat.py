import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class FaceMeshCompat:
    """Maintains the same interface as mp.solutions.face_mesh.FaceMesh."""

    def __init__(self, model_path="face_landmarker.task", max_num_faces=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 refine_landmarks=False):
        # refine_landmarks kept for API compat (Tasks API always outputs 478)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_rgb):
        """Equivalent to old FaceMesh.process()."""
        timestamp_ms = int(time.monotonic() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return _Result(result.face_landmarks)

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None


class _Result:
    """Mimics old FaceMesh result: has .multi_face_landmarks list or None."""

    def __init__(self, face_landmarks_list):
        if face_landmarks_list:
            self.multi_face_landmarks = [_LandmarkList(lms) for lms in face_landmarks_list]
        else:
            self.multi_face_landmarks = None


class _LandmarkList:
    """Mimics .landmark accessor returning list of NormalizedLandmark."""

    def __init__(self, landmarks):
        self.landmark = landmarks   # each has .x, .y, .z attributes