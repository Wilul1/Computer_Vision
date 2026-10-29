"""Rule-based activity classifier with temporal smoothing."""
from collections import Counter, deque

from . import config


def classify_frame(f: dict) -> str:
    """One frame's label from posture + speed features. Order matters: lying is
    checked first, then sitting, then the upright activities by speed."""
    if f["torso_angle"] >= config.LYING_TORSO_ANGLE_MIN:
        return "lying"

    knee, hip, thigh = f["knee_angle"], f["hip_angle"], f["thigh_ratio"]
    side_view_sit = knee is not None and hip is not None and knee < config.SIT_KNEE_ANGLE_MAX and hip < config.SIT_HIP_ANGLE_MAX
    front_view_sit = thigh is not None and thigh < config.SIT_THIGH_RATIO_MAX
    slow = f["hip_speed"] < config.WALK_HIP_SPEED_MIN and f["leg_speed"] < config.WALK_LEG_SPEED_MIN
    if (side_view_sit or front_view_sit) and slow:
        return "sitting"

    if f["hip_speed"] >= config.RUN_HIP_SPEED_MIN or f["leg_speed"] >= config.RUN_LEG_SPEED_MIN:
        return "running"
    if not slow:
        return "walking"
    return "standing"


class ActivityClassifier:
    """Majority vote over the last N frames removes one-frame flicker."""

    def __init__(self, window: int = config.SMOOTHING_WINDOW):
        self.recent = deque(maxlen=window)

    def reset(self):
        self.recent.clear()

    def __call__(self, features: dict) -> tuple[str, str, float]:
        raw = classify_frame(features)
        self.recent.append(raw)
        label, votes = Counter(self.recent).most_common(1)[0]
        return label, raw, votes / len(self.recent)
