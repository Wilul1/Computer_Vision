"""Preprocessing shared by the prototype and every evaluation script, so the
two can never drift apart."""
import cv2
import numpy as np

from . import config


def resize_max_side(frame: np.ndarray, max_side: int = config.PROCESS_MAX_SIDE) -> np.ndarray:
    """Downscale so the longer side is `max_side`. Smaller frames are left alone."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def enhance_contrast(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of Lab: brightens dark rooms without shifting colours."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def to_motion_input(frame: np.ndarray) -> np.ndarray:
    """Grayscale + Gaussian blur: removes sensor noise before background subtraction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = config.DENOISE_KERNEL
    return cv2.GaussianBlur(gray, (k, k), 0)


def preprocess(frame: np.ndarray, enhance: bool = True) -> dict:
    """Full preprocessing step. Returns every intermediate the later stages need."""
    resized = resize_max_side(frame)
    enhanced = enhance_contrast(resized) if enhance else resized
    return {
        "bgr": enhanced,                                   # for YOLO + drawing
        "rgb": cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),  # for MediaPipe
        "motion": to_motion_input(enhanced),               # for MOG2
    }
