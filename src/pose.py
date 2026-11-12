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
    def __init__(self, video: bool = True):
        """video=True tracks the person between frames (faster, steadier);
        video=False detects from scratch every call (used for crops, whose
        window moves from frame to frame)."""
        self.video = video
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def __call__(self, rgb: np.ndarray, timestamp_ms: int):
        """Returns a (33, 3) array of [x_px, y_px, visibility], or None if no person."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            ts = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing time
            self._last_ts = ts
            result = self.landmarker.detect_for_video(image, ts)
        else:
            result = self.landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array(
            [[lm.x * w, lm.y * h, lm.visibility] for lm in result.pose_landmarks[0]],
            dtype=np.float32,
        )

    def close(self):
        self.landmarker.close()


def pose_on_crop(estimator: PoseEstimator, rgb: np.ndarray, box, target_h: int, margin: float):
    """Run pose on the person's box, enlarged by `margin` and upscaled to
    `target_h` pixels tall, then map the landmarks back to frame coordinates.
    Small or distant people become big enough for the pose model to find."""
    import cv2

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    crop = rgb[cy1:cy2, cx1:cx2]
    scale = max(1.0, target_h / crop.shape[0])
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lm = estimator(crop, 0)
    if lm is None:
        return None
    lm[:, 0] = lm[:, 0] / scale + cx1
    lm[:, 1] = lm[:, 1] / scale + cy1
    return lm
