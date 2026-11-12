"""Draw the workflow diagram used in the report and slides.

    python scripts/draw_pipeline.py      # -> docs/figures/pipeline.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "pipeline.png"
STAGE = "#e8eef7"
MAIN = "#d7ecd9"
BASE = "#f6e3cf"
EDGE = "#3b4a5a"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=9.3, weight="bold", color="#1d2733")
    ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=8.0, color="#2c3946", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.3))


def main():
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box(ax, 0.1, 2.0, 1.9, 1.6, "Input", "webcam, phone video,\nvideo file or photo", STAGE)
    box(ax, 2.4, 2.0, 2.0, 1.6, "preprocess()", "resize (long side ≤ 640)\nCLAHE on Lab L\ngray + Gaussian 5×5", STAGE)
    box(ax, 4.8, 3.75, 2.3, 1.55, "infer(): pose", "MediaPipe Pose: 33 landmarks\n(on upscaled YOLO crop\nif person < 120 px)", MAIN)
    box(ax, 4.8, 2.05, 2.3, 1.5, "infer(): person + camera", "YOLO11n person boxes\nLK optical flow + RANSAC\n→ camera motion", MAIN)
    box(ax, 4.8, 0.3, 2.3, 1.5, "infer(): motion", "MOG2 background\nsubtraction → blobs", BASE)
    box(ax, 7.5, 3.0, 2.4, 2.3, "postprocess(): main", "knee, hip angles\nthigh ratio, torso tilt\nhip speed − camera speed\nleg swing, stride\n→ rules (tuned on val)", MAIN)
    box(ax, 7.5, 0.3, 2.4, 1.5, "postprocess(): baseline", "blob shape + speed\n→ rules (tuned on val)", BASE)
    box(ax, 10.3, 2.0, 2.6, 1.6, "smoothing + render()", "majority vote, persistence\n→ label, boxes, skeleton,\nFPS overlay + CSV row", STAGE)

    arrow(ax, 2.0, 2.8, 2.4, 2.8)
    for y in (4.5, 2.8, 1.05):
        arrow(ax, 4.4, 2.8, 4.8, y)
    arrow(ax, 7.1, 4.5, 7.5, 4.3)
    arrow(ax, 7.1, 2.8, 7.5, 3.6)
    arrow(ax, 7.1, 1.05, 7.5, 1.05)
    arrow(ax, 9.9, 4.1, 10.3, 3.2)
    arrow(ax, 9.9, 1.05, 10.3, 2.4)
    ax.text(0.1, 0.05, "green = main method   ·   orange = MOG2 baseline (computed on the same frames for comparison)",
            fontsize=8.5, color="#4a5866")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
