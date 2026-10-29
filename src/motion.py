"""Baseline motion detector: MOG2 background subtraction + contours."""
import cv2
import numpy as np

from . import config


class MotionDetector:
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=config.MOG2_DETECT_SHADOWS,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, gray: np.ndarray) -> dict:
        mask = self.subtractor.apply(gray)
        # Shadows are 127, real foreground is 255: keep only the foreground.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Opening removes speckle noise, closing fills holes inside the person.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        min_area = config.MOTION_MIN_AREA_RATIO * mask.shape[0] * mask.shape[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
        return {
            "mask": mask,
            "boxes": boxes,                                  # (x, y, w, h)
            "ratio": float(np.count_nonzero(mask)) / mask.size,
            "moving": len(boxes) > 0,
        }


class MotionEventCounter:
    """Turns per-frame 'moving' flags into discrete events, so the baseline can be
    compared with a hand count ("the person moved 4 times in this clip")."""

    def __init__(self):
        self.events = 0
        self._run = 0       # consecutive moving frames
        self._still = config.MOTION_EVENT_GAP_FRAMES
        self._in_event = False

    def update(self, moving: bool) -> int:
        if moving:
            self._run += 1
            self._still = 0
            if not self._in_event and self._run >= config.MOTION_EVENT_MIN_FRAMES:
                self._in_event = True
                self.events += 1
        else:
            self._run = 0
            self._still += 1
            if self._still >= config.MOTION_EVENT_GAP_FRAMES:
                self._in_event = False
        return self.events
