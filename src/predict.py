"""Prototype entry point.

    python src/predict.py --input data/test/walking/s05_walk_01.mp4 --output outputs/
    python src/predict.py --input path/to/photo.jpg --output outputs/
    python src/predict.py --input 0 --show            # live webcam (0 = first camera)

Writes an annotated video/image and a per-frame CSV log to --output.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import MonitorPipeline  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_FIELDS = ["frame", "time", "activity", "raw_activity", "confidence", "moving", "motion_ratio",
              "motion_events", "knee_angle", "hip_angle", "thigh_ratio", "torso_angle",
              "hip_speed", "leg_speed", "fps"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Motion detection and activity monitoring")
    p.add_argument("--input", required=True, help="image, video file, or camera index (0, 1, ...)")
    p.add_argument("--output", default="outputs/", help="folder for the annotated result and CSV log")
    p.add_argument("--show", action="store_true", help="show a live window (press q to stop)")
    p.add_argument("--no-yolo", action="store_true", help="skip YOLO person boxes (faster)")
    p.add_argument("--no-enhance", action="store_true", help="skip CLAHE contrast enhancement")
    p.add_argument("--max-seconds", type=float, default=None, help="stop after this much video")
    return p.parse_args(argv)


def predict_image(path: Path, out_dir: Path, pipe: MonitorPipeline) -> dict:
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    r = pipe.process(frame, 0.0)
    out_path = out_dir / f"{path.stem}_annotated.jpg"
    cv2.imwrite(str(out_path), r["annotated"])
    print(f"{path.name}: {r['activity']} -> {out_path}")
    return r


def predict_stream(source, name: str, out_dir: Path, pipe: MonitorPipeline, show: bool, max_seconds=None) -> list[dict]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
    is_camera = isinstance(source, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer, rows = None, []
    log_path = out_dir / f"{name}_log.csv"
    video_path = out_dir / f"{name}_annotated.mp4"
    wall_start = time.perf_counter()
    with open(log_path, "w", newline="") as f:
        log = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        log.writeheader()
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Camera: real elapsed time. File: the video's own timeline, so speeds
            # are correct even when processing is slower than real time.
            t = time.perf_counter() - wall_start if is_camera else idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break
            r = pipe.process(frame, t)
            if writer is None:
                h, w = r["annotated"].shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps if not is_camera else 15, (w, h))
            writer.write(r["annotated"])
            log.writerow({"frame": idx, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
            rows.append({k: v for k, v in r.items() if k not in ("annotated", "landmarks")})
            if show:
                cv2.imshow("Activity monitor (q to quit)", r["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    if rows:
        print(f"{name}: {idx} frames, avg FPS {sum(r['fps'] for r in rows) / len(rows):.1f}, "
              f"motion events {rows[-1]['motion_events']}")
        print(f"  video -> {video_path}\n  log   -> {log_path}")
    return rows


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = MonitorPipeline(use_yolo=not args.no_yolo, enhance=not args.no_enhance)
    try:
        if args.input.isdigit():
            predict_stream(int(args.input), f"camera{args.input}", out_dir, pipe, show=True, max_seconds=args.max_seconds)
        else:
            path = Path(args.input)
            if not path.exists():
                sys.exit(f"Input not found: {path}")
            if path.suffix.lower() in IMAGE_EXTS:
                predict_image(path, out_dir, pipe)
            else:
                predict_stream(str(path), path.stem, out_dir, pipe, args.show, args.max_seconds)
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
