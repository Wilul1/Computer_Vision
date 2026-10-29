"""Record labelled clips for the dataset from a webcam.

    python scripts/record_clips.py --activity walking --source s01-wilmark-room --seconds 10 --count 3

Saves to data/raw/<activity>/<source>__<n>.mp4. `--source` identifies ONE
recording session (same person + same place + same camera). All clips of one
source always end up in the same split, so test never contains near-copies of
training footage.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activity", required=True, choices=config.ACTIVITIES)
    p.add_argument("--source", required=True, help="session id, e.g. s01-cindy-dorm-webcam (no '__')")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--fps", type=float, default=30)
    args = p.parse_args()
    if "__" in args.source:
        sys.exit("--source must not contain '__'")

    out_dir = config.ROOT / "data" / "raw" / args.activity
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Camera {args.camera} not available")

    existing = len(list(out_dir.glob(f"{args.source}__*.mp4")))
    for n in range(existing + 1, existing + 1 + args.count):
        # 3 second countdown so the person can get into position.
        end = time.time() + 3
        while time.time() < end:
            ok, frame = cap.read()
            if ok:
                cv2.putText(frame, f"{args.activity} #{n} in {end - time.time():.0f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("record", frame)
                cv2.waitKey(1)
        path = out_dir / f"{args.source}__{n:02d}.mp4"
        writer, start = None, time.time()
        while time.time() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(frame)
            shown = frame.copy()
            cv2.circle(shown, (30, 30), 10, (0, 0, 255), -1)
            cv2.imshow("record", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer:
            writer.release()
            print(f"saved {path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
