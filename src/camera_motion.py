"""Estimate how much the CAMERA moved between frames, so a person the camera is
following (common in movies and hand-held phone video) is not read as standing.

Method: track corner points on the background with Lucas-Kanade optical flow,
ignoring points on the person, and take the median shift. The median is robust
to the few points that land on other moving things.
"""
import cv2
import numpy as np

MAX_CORNERS = 150
MIN_POINTS = 12          # fewer tracked points than this -> "unknown", assume no camera motion
PERSON_MARGIN = 0.15     # grow the person box by 15% before masking it out


class CameraMotion:
    def __init__(self):
        self.prev = None
        self.offset = np.zeros(2, np.float32)   # accumulated background shift (px)

    def reset(self):
        self.prev = None
        self.offset[:] = 0

    def update(self, gray: np.ndarray, person_box=None) -> np.ndarray:
        """gray: current frame. person_box: (x1, y1, x2, y2) to exclude, or None.
        Returns the accumulated background offset in pixels."""
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            return self.offset.copy()

        mask = np.full(gray.shape, 255, np.uint8)
        if person_box is not None:
            x1, y1, x2, y2 = person_box
            mx, my = (x2 - x1) * PERSON_MARGIN, (y2 - y1) * PERSON_MARGIN
            cv2.rectangle(mask, (int(x1 - mx), int(y1 - my)), (int(x2 + mx), int(y2 + my)), 0, -1)

        pts = cv2.goodFeaturesToTrack(self.prev, MAX_CORNERS, 0.01, 8, mask=mask)
        if pts is not None and len(pts) >= MIN_POINTS:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, pts, None, winSize=(15, 15), maxLevel=2)
            good = status.ravel() == 1
            if good.sum() >= MIN_POINTS:
                self.offset += np.median((nxt[good] - pts[good]).reshape(-1, 2), axis=0)
        self.prev = gray
        return self.offset.copy()
