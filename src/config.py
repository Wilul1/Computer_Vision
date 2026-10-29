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

# --- Rule-based activity classifier -------------------------------------------
# Angles in degrees; speeds in body-heights per second, so they do not depend on
# how far the person stands from the camera.
SIT_KNEE_ANGLE_MAX = 130        # knee bent more than this -> leg is folded
SIT_HIP_ANGLE_MAX = 140         # torso-thigh angle closed -> sitting posture
SIT_THIGH_RATIO_MAX = 0.45      # thigh nearly horizontal (front view of sitting)
LYING_TORSO_ANGLE_MIN = 60      # torso tilted more than this from vertical
WALK_HIP_SPEED_MIN = 0.25       # body travel speed that counts as walking
WALK_LEG_SPEED_MIN = 0.60       # ankle swing speed (walking toward the camera)
RUN_HIP_SPEED_MIN = 1.20
RUN_LEG_SPEED_MIN = 2.20
SPEED_WINDOW_SEC = 0.5          # speeds are averaged over this window
SMOOTHING_WINDOW = 9            # majority vote over the last N frame labels

ACTIVITIES = ["standing", "sitting", "walking", "running", "lying"]
