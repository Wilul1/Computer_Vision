"""Before/after figure for each preprocessing and feature step (required by the guide).

    python scripts/preprocess_strip.py --input data/val/walking/a.mp4 data/val/sitting/b.mp4 photo.jpg
    python scripts/preprocess_strip.py --input data/val --per-class 1      # one clip per activity folder

One row per sample, columns:
original | resized | CLAHE enhanced | gray + blur | MOG2 mask | pose + features
Saved to docs/figures/preprocessing.jpg (or --out).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.features import posture_features  # noqa: E402
from src.motion import MotionDetector  # noqa: E402
from src.pose import CONNECTIONS, PoseEstimator  # noqa: E402
from src.preprocess import enhance_contrast, resize_max_side, to_motion_input  # noqa: E402

TILE_W, TILE_H = 320, 240
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def tile(img, title):
    """Letterbox into a fixed-size tile so rows of different aspect ratios line up."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(TILE_W / w, TILE_H / h)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    out = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    y, x = (TILE_H - img.shape[0]) // 2, (TILE_W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.rectangle(out, (0, 0), (TILE_W, 22), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def load_sample(path: Path, frame_idx: int):
    """Returns (frame, MOG2 mask or None). For videos MOG2 is warmed up on earlier frames."""
    if path.suffix.lower() in IMAGE_EXTS:
        return cv2.imread(str(path)), None
    cap, det = cv2.VideoCapture(str(path)), MotionDetector()
    frame = mask = None
    for _ in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
        mask = det(to_motion_input(enhance_contrast(resize_max_side(f))))["mask"]
    cap.release()
    return frame, mask


def pose_tile(enhanced):
    pose = PoseEstimator()
    lm = pose(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), 0)
    pose.close()
    out = enhanced.copy()
    if lm is None:
        return tile(out, "pose: no person found")
    for a, b in CONNECTIONS:
        if lm[a, 2] >= config.LANDMARK_MIN_VISIBILITY and lm[b, 2] >= config.LANDMARK_MIN_VISIBILITY:
            cv2.line(out, tuple(map(int, lm[a, :2])), tuple(map(int, lm[b, :2])), (0, 255, 0), 2)
    f = posture_features(lm)
    fmt = lambda v, d=0: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    text = [f"knee {fmt(f['knee_angle'])}  hip {fmt(f['hip_angle'])}",
            f"thigh {fmt(f['thigh_ratio'], 2)}  torso {fmt(f['torso_angle'])}"]
    # Text is drawn before the tile shrinks the image, so scale it to stay readable.
    s = max(out.shape[0] / TILE_H, out.shape[1] / TILE_W)
    for i, t in enumerate(text):
        y = int(out.shape[0] - 8 * s - 20 * s * (len(text) - 1 - i))
        cv2.putText(out, t, (int(6 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), int(4 * s))
        cv2.putText(out, t, (int(6 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 255), max(1, int(s)))
    return tile(out, "pose landmarks + features")


def collect_inputs(inputs, per_class):
    paths = []
    for item in map(Path, inputs):
        if item.is_dir():
            for cls in sorted(p for p in item.iterdir() if p.is_dir()):
                clips = sorted(c for c in cls.iterdir() if c.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
                paths += clips[:per_class]
        else:
            paths.append(item)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="images, videos, or a split folder")
    p.add_argument("--frame", type=int, default=60, help="frame index used for videos")
    p.add_argument("--per-class", type=int, default=1, help="samples per activity when given a folder")
    p.add_argument("--out", default=str(config.ROOT / "docs" / "figures" / "preprocessing.jpg"))
    args = p.parse_args()

    rows = []
    for path in collect_inputs(args.input, args.per_class):
        frame, mask = load_sample(path, args.frame)
        if frame is None:
            print(f"skipped (unreadable): {path}")
            continue
        resized = resize_max_side(frame)
        enhanced = enhance_contrast(resized)
        name = f"{path.parent.name}/{path.name}"[:34]
        rows.append(np.hstack([
            tile(frame, f"{name}  {frame.shape[1]}x{frame.shape[0]}"),
            tile(resized, f"resized {resized.shape[1]}x{resized.shape[0]}"),
            tile(enhanced, "CLAHE on Lab L-channel"),
            tile(to_motion_input(enhanced), "gray + Gaussian 5x5"),
            tile(mask if mask is not None else np.zeros_like(resized), "MOG2 mask" if mask is not None else "MOG2: video only"),
            pose_tile(enhanced),
        ]))
    if not rows:
        sys.exit("No readable inputs.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"saved {out} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
