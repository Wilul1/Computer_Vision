"""Geometric features from pose landmarks: joint angles, body proportions and
movement speeds. Everything is scale-free (angles or ratios to body size) so the
same thresholds work near and far from the camera."""
from collections import deque

import numpy as np

from . import config
from .pose import L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER


def angle(a, b, c) -> float:
    """Angle ABC in degrees (at point b)."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _visible(lm, *idx) -> bool:
    return all(lm[i, 2] >= config.LANDMARK_MIN_VISIBILITY for i in idx)


def _side_mean(lm, left: tuple, right: tuple, fn):
    """Average a measurement over the body sides whose joints are visible."""
    vals = [fn(*(lm[i, :2] for i in side)) for side in (left, right) if _visible(lm, *side)]
    return float(np.mean(vals)) if vals else None


def posture_features(lm: np.ndarray) -> dict:
    """Features from a single frame (no time needed)."""
    sh = (lm[L_SHOULDER, :2] + lm[R_SHOULDER, :2]) / 2
    hip = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2
    torso = np.linalg.norm(sh - hip) + 1e-6

    knee_angle = _side_mean(lm, (L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE), angle)
    hip_angle = _side_mean(lm, (L_SHOULDER, L_HIP, L_KNEE), (R_SHOULDER, R_HIP, R_KNEE), angle)
    # Vertical drop from hip to knee relative to torso length. Standing ~1.0,
    # sitting facing the camera ~0.0-0.4 because the thigh points at the lens.
    thigh_ratio = _side_mean(lm, (L_HIP, L_KNEE), (R_HIP, R_KNEE), lambda h, k: abs(k[1] - h[1]) / torso)
    # Torso tilt from vertical: 0 = upright, 90 = horizontal (lying).
    dx, dy = sh - hip
    torso_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    # Body height: shoulders to ankles if visible, else 2.5 torso lengths.
    if _visible(lm, L_ANKLE, R_ANKLE):
        body_h = float(np.linalg.norm(sh - (lm[L_ANKLE, :2] + lm[R_ANKLE, :2]) / 2))
    else:
        body_h = 2.5 * torso
    return {
        # Off-screen joints still get guessed coordinates from MediaPipe, but with
        # low visibility. Without hips and at least one thigh in view, posture rules
        # would only be guessing (e.g. someone seated at a desk looks "upright").
        "hips_visible": _visible(lm, L_HIP) or _visible(lm, R_HIP),
        "legs_visible": thigh_ratio is not None,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "thigh_ratio": thigh_ratio,
        "torso_angle": torso_angle,
        "body_h": max(body_h, torso),
        "mid_hip": hip,
        "ankles_rel": (lm[[L_ANKLE, R_ANKLE], :2] - hip) if _visible(lm, L_ANKLE, R_ANKLE) else None,
    }


class MotionFeatures:
    """Keeps a short history of the body and turns it into speeds (body-heights / s)."""

    def __init__(self):
        self.history = deque()

    def reset(self):
        self.history.clear()

    def update(self, t: float, feats: dict) -> dict:
        self.history.append((t, feats))
        while self.history and t - self.history[0][0] > config.SPEED_WINDOW_SEC:
            self.history.popleft()
        if len(self.history) < 3:
            return {"hip_speed": 0.0, "leg_speed": 0.0}

        (t0, f0), (t1, f1) = self.history[0], self.history[-1]
        dt = max(t1 - t0, 1e-3)
        scale = np.median([f["body_h"] for _, f in self.history])
        # Horizontal travel of the hips across the frame.
        hip_speed = abs(f1["mid_hip"][0] - f0["mid_hip"][0]) / scale / dt
        # Leg swing: range of ankle positions relative to the hips over the window.
        # Catches walking/running toward the camera where hips barely shift. Using
        # the range (not the summed path) keeps landmark jitter from adding up.
        ankles = [f["ankles_rel"] for _, f in self.history if f["ankles_rel"] is not None]
        if len(ankles) >= 3:
            a = np.stack(ankles)
            leg_speed = float((a.max(axis=0) - a.min(axis=0)).sum()) / scale / dt
        else:
            leg_speed = 0.0
        return {"hip_speed": float(hip_speed), "leg_speed": float(leg_speed)}
