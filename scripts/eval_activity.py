"""Evaluate the rule-based activity classifier on a split (default: val).

Each clip lives in a folder named after its true activity. A clip's prediction
is the most common smoothed label over frames where a person was found.
Reports accuracy, macro-F1, a confusion matrix and FPS.

    python scripts/eval_activity.py --split val
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.pipeline import MonitorPipeline  # noqa: E402
from src.predict import predict_stream  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--no-yolo", action="store_true")
    args = p.parse_args()
    if args.split == "test":
        print("WARNING: test split is for the final evaluation only. Do not tune on these numbers.")

    split_dir = config.ROOT / "data" / args.split
    clips = sorted(c for c in split_dir.rglob("*") if c.suffix.lower() in VIDEO_EXTS)
    if not clips:
        sys.exit(f"No clips in {split_dir}. Run scripts/make_splits.py first.")

    out_dir = config.OUTPUTS_DIR / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, fps, per_clip = [], [], [], []
    for clip in clips:
        pipe = MonitorPipeline(use_yolo=not args.no_yolo)  # fresh state per clip
        rows = predict_stream(str(clip), f"{clip.parent.name}__{clip.stem}", out_dir, pipe, show=False)
        pipe.close()
        labels = [r["activity"] for r in rows if r["activity"] != "no person"]
        pred = Counter(labels).most_common(1)[0][0] if labels else "no person"
        y_true.append(clip.parent.name)
        y_pred.append(pred)
        clip_fps = sum(r["fps"] for r in rows) / max(len(rows), 1)
        fps.append(clip_fps)
        frame_acc = sum(l == clip.parent.name for l in labels) / max(len(labels), 1)
        per_clip.append({"clip": str(clip.relative_to(config.ROOT)), "true": clip.parent.name,
                         "pred": pred, "frame_accuracy": round(frame_acc, 3), "fps": round(clip_fps, 1)})
        print(f"  {'OK ' if pred == clip.parent.name else 'XX '} {clip.parent.name:9s} -> {pred:9s} "
              f"(frames correct {frame_acc:.0%})")

    labels = sorted(set(y_true))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    f1 = macro_f1(y_true, y_pred, labels)
    pred_labels = sorted(set(y_true) | set(y_pred))
    confusion = {t: {p: sum(a == t and b == p for a, b in zip(y_true, y_pred)) for p in pred_labels} for t in labels}

    print(f"\n{args.split}: {len(clips)} clips  accuracy {acc:.3f}  macro-F1 {f1:.3f}  "
          f"mean FPS {sum(fps) / len(fps):.1f}")
    print("confusion (rows = true, cols = predicted):")
    print("           " + " ".join(f"{p[:8]:>8s}" for p in pred_labels))
    for t in labels:
        print(f"  {t:9s}" + " ".join(f"{confusion[t][p]:8d}" for p in pred_labels))

    metrics = {"split": args.split, "clips": len(clips), "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
               "mean_fps": round(sum(fps) / len(fps), 2), "confusion": confusion, "per_clip": per_clip}
    metrics_path = config.OUTPUTS_DIR / "metrics" / f"activity_{args.split}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
