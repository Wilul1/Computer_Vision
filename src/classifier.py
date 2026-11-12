"""Activity rules for the main method (pose) and the baseline (MOG2 blob),
plus the temporal post-processing both share."""
from collections import Counter, deque

from .params import Params

NO_PERSON, UNKNOWN = "no person", "unknown"
IGNORED = (NO_PERSON, UNKNOWN)   # labels that never win a vote or count as a prediction


def classify_pose(f: dict, p: Params) -> str:
    """One frame's label from pose features. Order matters: visibility first,
    then lying, then sitting, then the upright activities by speed."""
    if not f["hips_visible"]:
        return UNKNOWN
    if f["torso_angle"] >= p.lying_torso_angle_min:
        return "lying"
    if not f["legs_visible"]:
        return UNKNOWN

    hip_speed = f["hip_speed_comp"] if p.camera_compensation else f["hip_speed"]
    knee, hip, thigh = f["knee_angle"], f["hip_angle"], f["thigh_ratio"]
    side_view_sit = knee is not None and hip is not None and knee < p.sit_knee_angle_max and hip < p.sit_hip_angle_max
    front_view_sit = thigh is not None and thigh < p.sit_thigh_ratio_max
    # Leg swing only counts if it is a real stride (ankles moving apart and
    # together), not both feet jiggling in place. walk_stride_min = 0 disables
    # the check (Milestone 2/3 behaviour).
    striding = f.get("stride", 0.0) >= p.walk_stride_min
    leg_walk = f["leg_speed"] >= p.walk_leg_speed_min and striding
    leg_run = f["leg_speed"] >= p.run_leg_speed_min and striding
    hips_still = hip_speed < p.walk_hip_speed_min
    slow = hips_still and not leg_walk
    # seated_first: a folded leg is sitting as long as the body does not travel,
    # whatever the feet are doing (Milestone 4 fix for 11 test failures).
    if (side_view_sit or front_view_sit) and (hips_still if p.seated_first else slow):
        return "sitting"

    if hip_speed >= p.run_hip_speed_min or leg_run:
        return "running"
    if not slow:
        return "walking"
    return "standing"


def classify_blob(b: dict | None, p: Params) -> str:
    """Baseline: shape and speed of the largest MOG2 foreground blob only. No
    pose, no person detector. A still person fades into the MOG2 background,
    so 'no blob' is read as standing still."""
    if b is None:
        return "standing"
    if b["aspect"] < p.base_lying_aspect_max:
        return "lying"
    if b["aspect"] < p.base_upright_aspect_min:
        return "sitting"
    if b["speed"] >= p.base_run_speed_min:
        return "running"
    if b["speed"] >= p.base_walk_speed_min:
        return "walking"
    return "standing"


class TemporalFilter:
    """Post-processing: majority vote over a window, then persistence: the shown
    label only changes once the new label has won `persist_frames` votes in a
    row (more for sitting, which a running stride can briefly look like)."""

    def __init__(self, p: Params, method: str = "main"):
        if method == "baseline":
            window, self.persist, self.sit_persist = p.base_smoothing_window, p.base_persist_frames, p.base_persist_frames
        else:
            window, self.persist, self.sit_persist = p.smoothing_window, p.persist_frames, p.sit_persist_frames
        self.recent = deque(maxlen=max(1, int(window)))
        self.current = None
        self.candidate, self.streak = None, 0

    def __call__(self, raw: str) -> tuple[str, float]:
        if raw in IGNORED:
            # Nothing reliable this frame: keep showing the last label if we have one.
            return (self.current or raw), 0.0
        self.recent.append(raw)
        winner, votes = Counter(self.recent).most_common(1)[0]
        conf = votes / len(self.recent)
        if self.current is None:
            self.current = winner
        elif winner != self.current:
            self.streak = self.streak + 1 if winner == self.candidate else 1
            self.candidate = winner
            need = self.sit_persist if winner == "sitting" else self.persist
            if self.streak >= need:
                self.current, self.candidate, self.streak = winner, None, 0
        else:
            self.candidate, self.streak = None, 0
        return self.current, conf


def replay(trace: list[dict], p: Params, method: str = "main") -> list[str]:
    """Re-run the rules + temporal filter over cached per-frame features. This is
    how tune.py and evaluate.py score many parameter sets without re-running
    the neural networks."""
    filt = TemporalFilter(p, method)
    out = []
    for f in trace:
        if method == "baseline":
            raw = classify_blob(f["blob"], p)
        elif f["pose"] is None or (p.require_person_box and f.get("n_people", -1) == 0):
            raw = NO_PERSON
        else:
            raw = classify_pose(f["pose"], p)
        out.append(filt(raw)[0])
    return out
