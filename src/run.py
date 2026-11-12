"""One pipeline, one command: raw input -> visible output, no manual steps.

    python src/run.py --input data/test/ --output outputs/run_2026-11-12/   # every image/video in a folder
    python src/run.py --input demo.mp4 --output outputs/demo_annotated.mp4    # one video
    python src/run.py --input photo.jpg --output outputs/                      # one image
    python src/run.py --webcam 0                                               # live camera, q to quit

For each input it writes an annotated video/image and a per-frame CSV
(activity, baseline activity, angles, speeds, FPS). A folder run also writes
summary.csv with one row per input. Uses the parameters tuned on the val split
(configs/tuned_params.json); pass --params to use another file.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifier import IGNORED  # noqa: E402
from src.params import Params  # noqa: E402
from src.pipeline import MonitorPipeline  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LOG_FIELDS = ["frame", "time", "activity", "activity_baseline", "raw_activity", "confidence", "moving",
              "motion_ratio", "motion_events", "knee_angle", "hip_angle", "thigh_ratio", "torso_angle",
              "hip_speed", "hip_speed_comp", "leg_speed", "fps"]


def majority(labels):
    labels = [l for l in labels if l not in IGNORED]
    return Counter(labels).most_common(1)[0][0] if labels else "no person"


def run_image(path: Path, out_path: Path, pipe: MonitorPipeline) -> dict:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cannot read image {path}")
    r = pipe.process(frame, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), r["annotated"])
    return {"frames": 1, "activity": r["activity"], "activity_baseline": r["activity_baseline"],
            "fps": r["fps"], "output": out_path, "trace": [r["trace"]], "timings": [r["timing_ms"]]}


def run_stream(source, out_video: Path | None, out_csv: Path | None, pipe: MonitorPipeline,
               show: bool = False, max_seconds: float | None = None) -> dict:
    """Video file or camera index -> annotated video + per-frame CSV."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"cannot open video source {source}")
    is_camera = isinstance(source, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, rows, trace, timings = None, [], [], []
    log_file = log = None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(out_csv, "w", newline="")
        log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS, extrasaction="ignore")
        log.writeheader()
    wall_start, idx = time.perf_counter(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Camera: real elapsed time. File: the video's own clock, so speeds are
            # right even when processing is slower than real time.
            t = time.perf_counter() - wall_start if is_camera else idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break
            r = pipe.process(frame, t)
            if out_video is not None:
                if writer is None:
                    out_video.parent.mkdir(parents=True, exist_ok=True)
                    h, w = r["annotated"].shape[:2]
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             15 if is_camera else src_fps, (w, h))
                writer.write(r["annotated"])
            if log:
                log.writerow({"frame": idx, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
            rows.append((r["activity"], r["activity_baseline"], r["fps"]))
            trace.append(r["trace"])
            timings.append(r["timing_ms"])
            if show:
                cv2.imshow("Activity monitor (q to quit)", r["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if show:
            cv2.destroyAllWindows()
    return {"frames": idx, "fps_source": src_fps,
            "activity": majority(a for a, _, _ in rows), "activity_baseline": majority(b for _, b, _ in rows),
            "fps": sum(f for _, _, f in rows) / max(len(rows), 1), "output": out_video,
            "trace": trace, "timings": timings}


def collect_inputs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Motion detection and activity monitoring")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="image, video, or folder of images/videos")
    src.add_argument("--webcam", type=int, help="camera index (0 = built-in webcam)")
    ap.add_argument("--output", default="outputs/", help="output folder, or a .mp4/.jpg file for a single input")
    ap.add_argument("--method", choices=["main", "baseline"], default="main", help="which label to show")
    ap.add_argument("--params", default=None, help="parameter file (default: configs/tuned_params.json)")
    ap.add_argument("--show", action="store_true", help="live window while processing files")
    ap.add_argument("--no-yolo", action="store_true", help="skip YOLO person boxes (faster)")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    params = Params.load(args.params)
    pipe = MonitorPipeline(params, method=args.method, use_yolo=not args.no_yolo)
    out = Path(args.output)
    try:
        if args.webcam is not None:
            out_dir = out if out.suffix == "" else out.parent
            res = run_stream(args.webcam, out_dir / f"webcam{args.webcam}_annotated.mp4",
                             out_dir / f"webcam{args.webcam}_log.csv", pipe, show=True, max_seconds=args.max_seconds)
            print(f"webcam: {res['frames']} frames, {res['fps']:.1f} FPS -> {res['output']}")
            return

        inp = Path(args.input)
        if not inp.exists():
            sys.exit(f"Input not found: {inp}")
        inputs = collect_inputs(inp) if inp.is_dir() else [inp]
        if not inputs:
            sys.exit(f"No images or videos in {inp}")
        single_file_out = len(inputs) == 1 and out.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
        summary = []
        for i, path in enumerate(inputs, 1):
            pipe.reset()
            rel = path.relative_to(inp) if inp.is_dir() else Path(path.name)
            is_image = path.suffix.lower() in IMAGE_EXTS
            if single_file_out:
                target = out
            else:
                target = out / rel.parent / f"{path.stem}_annotated{'.jpg' if is_image else '.mp4'}"
            try:
                if is_image:
                    res = run_image(path, target, pipe)
                else:
                    res = run_stream(str(path), target, target.with_name(f"{path.stem}_log.csv"), pipe,
                                     show=args.show, max_seconds=args.max_seconds)
            except ValueError as e:
                print(f"[{i}/{len(inputs)}] SKIPPED {path}: {e}")
                continue
            print(f"[{i}/{len(inputs)}] {rel}: {res['activity']} (baseline: {res['activity_baseline']}), "
                  f"{res['frames']} frames, {res['fps']:.1f} FPS -> {res['output']}")
            summary.append({"input": str(rel), "frames": res["frames"], "activity": res["activity"],
                            "activity_baseline": res["activity_baseline"], "fps": round(res["fps"], 1),
                            "output": str(res["output"])})
        if inp.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0]))
                w.writeheader()
                w.writerows(summary)
            print(f"summary -> {out / 'summary.csv'}")
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
