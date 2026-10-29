"""MediaPipe Pose Landmarker wrapper: 33 body landmarks per frame."""
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config

# Landmark indices used by the feature extractor.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Skeleton edges for drawing (subset of the 33 points, face omitted).
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32),
]


def ensure_model() -> str:
    path = config.POSE_MODEL_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model to {path} ...")
        urllib.request.urlretrieve(config.POSE_MODEL_URL, path)
    return str(path)


class PoseEstimator:
    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def __call__(self, rgb: np.ndarray, timestamp_ms: int):
        """Returns a (33, 3) array of [x_px, y_px, visibility], or None if no person."""
        ts = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing time
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array(
            [[lm.x * w, lm.y * h, lm.visibility] for lm in result.pose_landmarks[0]],
            dtype=np.float32,
        )

    def close(self):
        self.landmarker.close()
