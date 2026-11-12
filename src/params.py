"""Classifier parameters. These are tuned on the VAL split by scripts/tune.py and
saved to configs/tuned_params.json; run.py and evaluate.py load that file.

The defaults are the hand-set values from Milestone 2 ("v1"), kept so the
M2 version can still be re-scored as a comparison.
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import config

TUNED_PATH = config.ROOT / "configs" / "tuned_params.json"


@dataclass
class Params:
    # --- Main method: pose rules (angles in degrees, speeds in body-heights/s) ---
    sit_knee_angle_max: float = 130
    sit_hip_angle_max: float = 140
    sit_thigh_ratio_max: float = 0.45
    lying_torso_angle_min: float = 60
    walk_hip_speed_min: float = 0.25
    walk_leg_speed_min: float = 0.60
    run_hip_speed_min: float = 1.20
    run_leg_speed_min: float = 2.20
    # Subtract the camera's own movement (background optical flow) from the
    # hip speed. False reproduces the Milestone 2 behaviour.
    camera_compensation: bool = False
    # Milestone 4 fixes for "limb motion in place read as walking" (defaults =
    # off, so the M2/M3 parameter sets still score exactly as before):
    seated_first: bool = False          # a folded-leg posture is sitting unless the HIPS travel
    walk_stride_min: float = 0.0        # leg swing only counts as walking if the ankle gap
                                        # also changes this much (body-heights)
    # Milestone 4: trust a pose only if YOLO also sees a person in the frame
    # (test case TC13: MediaPipe drew a skeleton on a flower-patterned curtain).
    require_person_box: bool = False

    # --- Baseline: MOG2 foreground blob rules ---------------------------------
    # Blob aspect = height / width of the largest moving blob's bounding box.
    base_upright_aspect_min: float = 1.6
    base_lying_aspect_max: float = 0.9
    base_walk_speed_min: float = 0.25   # blob centre speed in blob-heights/s
    base_run_speed_min: float = 1.20

    # --- Temporal post-processing (each method has its own, tuned separately) --
    smoothing_window: int = 9           # majority vote over the last N frames
    persist_frames: int = 1             # a new label must win this many frames in a row
    sit_persist_frames: int = 1         # persistence before switching TO sitting
    base_smoothing_window: int = 9
    base_persist_frames: int = 1

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Params":
        """Tuned values if configs/tuned_params.json exists, else the M2 defaults."""
        path = Path(path) if path else TUNED_PATH
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.get("params", data).items() if k in known})

    def save(self, path: Path | str = TUNED_PATH, **meta):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**meta, "params": asdict(self)}, indent=2))
