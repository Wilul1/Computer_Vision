"""Before/after strip for each preprocessing step (required by the guide).

    python scripts/preprocess_strip.py --input data/val/walking/xxx.mp4 --frame 60
    python scripts/preprocess_strip.py --input some_photo.jpg

Saves docs/figures/preprocess_<name>.jpg with columns:
original | resized | CLAHE enhanced | gray + blur | MOG2 mask (videos only)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.motion import MotionDetector  # noqa: E402
from src.preprocess import enhance_contrast, resize_max_side, to_motion_input  # noqa: E402

TILE_H = 240


def tile(img, title):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * TILE_H / h), TILE_H))
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--frame", type=int, default=60, help="frame index for videos")
    args = p.parse_args()
    path = Path(args.input)

    mask = None
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(path))
    else:
        cap, det = cv2.VideoCapture(str(path)), MotionDetector()
        frame = None
        for _ in range(args.frame + 1):  # warm up MOG2 on the frames before
            ok, f = cap.read()
            if not ok:
                break
            frame = f
            mask = det(to_motion_input(enhance_contrast(resize_max_side(f))))["mask"]
        cap.release()
    if frame is None:
        sys.exit(f"Could not read {path}")

    resized = resize_max_side(frame)
    enhanced = enhance_contrast(resized)
    tiles = [tile(frame, f"original {frame.shape[1]}x{frame.shape[0]}"),
             tile(resized, f"resized {resized.shape[1]}x{resized.shape[0]}"),
             tile(enhanced, "CLAHE (Lab L-channel)"),
             tile(to_motion_input(enhanced), "gray + Gaussian blur")]
    if mask is not None:
        tiles.append(tile(mask, "MOG2 foreground mask"))
    out = config.ROOT / "docs" / "figures" / f"preprocess_{path.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
