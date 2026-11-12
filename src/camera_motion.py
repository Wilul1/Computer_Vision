"""Estimate how much the CAMERA moved between frames, so a person the camera is
following (common in movies and hand-held phone video) is not read as standing.

Method: track corner points on the background with Lucas-Kanade optical flow,
ignoring points on the person, then fit the camera's motion (shift + rotation +
zoom) with RANSAC, which ignores points that land on other moving things.
Milestone 3 used the plain median shift; on val it changed only 1 of 108
segments, which is why it was replaced.
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
                a, b = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
                # Milestone 4: fit translation + rotation + zoom with RANSAC, so
                # points on other moving things (and a zooming camera) are
                # rejected as outliers instead of pulling a median around.
                m, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if m is not None and inliers is not None and inliers.sum() >= MIN_POINTS:
                    # Shift of the frame centre under the fitted motion.
                    c = np.array([gray.shape[1] / 2, gray.shape[0] / 2], np.float32)
                    self.offset += (m[:, :2] @ c + m[:, 2]) - c
                else:
                    self.offset += np.median(b - a, axis=0)
        self.prev = gray
        return self.offset.copy()
