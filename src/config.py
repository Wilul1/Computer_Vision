"""Central settings. Every threshold the classifier uses lives here so it can be
tuned on the validation split (never on test) and explained in the defense."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# --- Preprocessing -----------------------------------------------------------
# Frames are resized so the longer side is this many pixels. 640 matches the
# YOLO input size and keeps CPU processing near real time.
PROCESS_MAX_SIDE = 640
CLAHE_CLIP_LIMIT = 2.0          # contrast boost for dark scenes (L channel of Lab)
CLAHE_TILE_GRID = (8, 8)
DENOISE_KERNEL = 5              # Gaussian blur kernel for the motion mask

# --- Motion (MOG2 background subtraction) -------------------------------------
MOG2_HISTORY = 300              # frames remembered as "background"
MOG2_VAR_THRESHOLD = 25         # higher = less sensitive to small pixel changes
MOG2_DETECT_SHADOWS = True      # shadows are marked 127 and then removed
MOTION_MIN_AREA_RATIO = 0.004   # ignore blobs smaller than 0.4% of the frame
MOTION_EVENT_MIN_FRAMES = 5     # motion must last this many frames to count as an event
MOTION_EVENT_GAP_FRAMES = 15    # still frames needed before a new event can start

# --- Person detection (YOLO) -------------------------------------------------
YOLO_WEIGHTS = "yolo11n.pt"     # nano model, CPU friendly, auto-downloaded by ultralytics
YOLO_CONF = 0.4
YOLO_EVERY_N_FRAMES = 3         # run YOLO every N frames and reuse boxes in between

# --- Pose (MediaPipe Pose Landmarker) ----------------------------------------
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_MIN_CONFIDENCE = 0.5
LANDMARK_MIN_VISIBILITY = 0.5
# Milestone 4: if YOLO's person box is shorter than this, or the full-frame pose
# finds nobody while YOLO does, run pose on the box crop upscaled to CROP_TARGET_H.
POSE_CROP_BELOW_PX = 120
POSE_CROP_TARGET_H = 256
POSE_CROP_MARGIN = 0.25

# --- Rule-based activity classifier -------------------------------------------
# The rule thresholds are tunable parameters: see src/params.py (defaults = the
# Milestone 2 values) and configs/tuned_params.json (tuned on val by scripts/tune.py).
# Speeds are in body-heights per second, so they do not depend on distance.
SPEED_WINDOW_SEC = 0.5          # speeds are measured over this window (fixed, not tuned)

ACTIVITIES = ["standing", "sitting", "walking", "running", "lying"]
