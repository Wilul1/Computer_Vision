"""Motion baseline required by the checkpoint guide: MOG2 + contours, compared
with a hand count on 2-3 clips, plus FPS.

Hand counts go in data/annotations/motion_counts.csv:
    clip,hand_count
    data/val/walking/s03-zheniel-hall__01.mp4,3

A "motion event" = the person starts moving after being still for
MOTION_EVENT_GAP_FRAMES frames (see src/config.py).

    python scripts/eval_motion_baseline.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.motion import MotionDetector, MotionEventCounter  # noqa: E402
from src.preprocess import preprocess  # noqa: E402


def count_events(path: Path) -> tuple[int, float, int]:
    cap = cv2.VideoCapture(str(path))
    det, counter = MotionDetector(), MotionEventCounter()
    frames, start = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        counter.update(det(preprocess(frame)["motion"])["moving"])
        frames += 1
    cap.release()
    return counter.events, frames / max(time.perf_counter() - start, 1e-6), frames


def main():
    counts_file = config.ROOT / "data" / "annotations" / "motion_counts.csv"
    if not counts_file.exists():
        sys.exit(f"Missing {counts_file}. Add 2-3 clips with hand-counted motion events.")
    with open(counts_file) as f:
        rows = list(csv.DictReader(f))

    results = []
    print(f"{'clip':55s} {'hand':>5s} {'MOG2':>5s} {'err':>4s} {'FPS':>6s}")
    for row in rows:
        path = config.ROOT / row["clip"]
        if not path.exists():
            print(f"  missing {path}")
            continue
        events, fps, frames = count_events(path)
        hand = int(row["hand_count"])
        results.append({"clip": row["clip"], "hand_count": hand, "detected": events,
                        "abs_error": abs(events - hand), "fps": round(fps, 1), "frames": frames})
        print(f"{row['clip'][:55]:55s} {hand:5d} {events:5d} {abs(events - hand):4d} {fps:6.1f}")

    if not results:
        sys.exit("No clips evaluated.")
    total_hand = sum(r["hand_count"] for r in results)
    mae = sum(r["abs_error"] for r in results) / len(results)
    summary = {
        "clips": len(results),
        "mean_abs_error": round(mae, 3),
        "count_accuracy": round(1 - sum(r["abs_error"] for r in results) / max(total_hand, 1), 3),
        "mean_fps": round(sum(r["fps"] for r in results) / len(results), 1),
        "per_clip": results,
    }
    print(f"\nMAE {summary['mean_abs_error']}  count accuracy {summary['count_accuracy']:.1%}  "
          f"mean FPS {summary['mean_fps']}")
    out = config.OUTPUTS_DIR / "metrics" / "motion_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
