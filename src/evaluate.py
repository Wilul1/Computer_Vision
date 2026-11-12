"""Reproduce the results table from the tag.

    python src/evaluate.py --split test          # the reported numbers (held-out test set)
    python src/evaluate.py --split val           # what tuning looked at

Ground truth is a list of SEGMENTS: a clip in data/<split>/<activity>/ is one
segment with its folder's label; a video in data/<split>/_segments/ has several
labelled time ranges listed in data/annotations/segments.csv. A segment's
prediction is the majority label over its frames (frames with no person or
hidden legs do not vote). Segments shorter than MIN_SEGMENT_SEC are skipped.

Every clip goes through the full pipeline ONCE (YOLO + pose + MOG2, timed after
a warm-up). Its per-frame features are cached in outputs/cache/<split>/, and
the three methods are scored on exactly the same frames:
  baseline  MOG2 blob shape + speed, rules tuned on val
  main_m2   pose rules with the Milestone 2 hand-set values (no tuning)
  main      pose rules + camera compensation + persistence, tuned on val

Outputs: outputs/metrics/eval_<split>.json, outputs/metrics/test_log_<split>.csv,
docs/figures/confusion_<split>_<method>.png, docs/results_<split>.md.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.classifier import IGNORED, NO_PERSON, replay  # noqa: E402
from src.params import Params  # noqa: E402

ROOT = config.ROOT
DATA = ROOT / "data"
METRICS = ROOT / "outputs" / "metrics"
CACHE = ROOT / "outputs" / "cache"
FIGURES = ROOT / "docs" / "figures"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MIN_SEGMENT_SEC = 1.0
WARMUP_FRAMES = 30          # timing ignores the first frames of each clip (model warm-up)
CACHE_VERSION = 3           # bump when the trace format changes
CLASSES = ["standing", "sitting", "walking", "running", "lying"]
METHODS = {"baseline": "MOG2 blob + rules (baseline)",
           "main_m2": "Pose + rules, M2 hand-set values",
           "main": "Pose + rules, tuned + camera compensation (main)"}


# ---------------------------------------------------------------- ground truth
def load_segments(split: str) -> list[dict]:
    seg_rows = {}
    seg_file = DATA / "annotations" / "segments.csv"
    if seg_file.exists():
        with open(seg_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seg_rows.setdefault(row["clip"], []).append(row)
    info = {}
    info_file = DATA / "annotations" / "clip_info.csv"
    if info_file.exists():
        with open(info_file, newline="", encoding="utf-8") as f:
            info = {row["clip"]: row for row in csv.DictReader(f)}

    segments = []
    for clip in sorted(p for p in (DATA / split).rglob("*") if p.suffix.lower() in VIDEO_EXTS):
        cond = info.get(f"{clip.parent.name}/{clip.name}", {}).get("condition", "own recording" if clip.name[:1] == "s" else "")
        if clip.parent.name == "_segments":
            rows = seg_rows.get(clip.name, [])
            for r in rows:
                start, end = float(r["start"]), float(r["end"])
                if end - start < MIN_SEGMENT_SEC:
                    continue
                # Frames that also fall inside a segment with a DIFFERENT label are
                # ambiguous (transitions); they are excluded from this segment.
                others = [(float(o["start"]), float(o["end"])) for o in rows if o["label"] != r["label"]]
                segments.append({"id": r["id"], "clip": clip, "start": start, "end": end,
                                 "label": r["label"], "exclude": others, "condition": cond})
        else:
            segments.append({"id": clip.stem, "clip": clip, "start": 0.0, "end": float("inf"),
                             "label": clip.parent.name, "exclude": [], "condition": cond})
    return segments


# ---------------------------------------------------------------- traces
def cache_path(split: str, clip: Path) -> Path:
    return CACHE / split / f"{clip.parent.name}__{clip.stem}.json"


def get_traces(split: str, clips: list[Path], rerun: bool, use_yolo: bool) -> dict:
    """Run the pipeline on every clip (or load the cached run)."""
    from src.pipeline import MonitorPipeline, to_jsonable
    from src.run import run_stream

    traces, pipe = {}, None
    todo = [c for c in clips if rerun or not cache_path(split, c).exists()
            or json.loads(cache_path(split, c).read_text()).get("version") != CACHE_VERSION]
    for i, clip in enumerate(clips, 1):
        cp = cache_path(split, clip)
        if clip in todo:
            if pipe is None:
                pipe = MonitorPipeline(Params(), use_yolo=use_yolo)
            pipe.reset()
            t0 = time.perf_counter()
            res = run_stream(str(clip), None, None, pipe)
            data = {"version": CACHE_VERSION, "clip": str(clip.relative_to(ROOT)), "yolo": use_yolo,
                    "fps_source": res["fps_source"], "trace": to_jsonable(res["trace"]), "timings": res["timings"]}
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data))
            print(f"  [{i}/{len(clips)}] ran {clip.name}: {res['frames']} frames in {time.perf_counter() - t0:.1f}s",
                  flush=True)
        traces[clip] = json.loads(cp.read_text())
    if pipe is not None:
        pipe.close()
    return traces


# ---------------------------------------------------------------- scoring
def segment_frames(seg: dict, trace: list[dict]) -> list[int]:
    idx = []
    for i, f in enumerate(trace):
        t = f["t"]
        if seg["start"] <= t <= seg["end"] and not any(s <= t <= e for s, e in seg["exclude"]):
            idx.append(i)
    return idx


def predict_segments(segments, traces, params: Params, method: str) -> list[dict]:
    labels_by_clip = {clip: replay(d["trace"], params, "baseline" if method == "baseline" else "main")
                      for clip, d in traces.items()}
    out = []
    for seg in segments:
        trace = traces[seg["clip"]]["trace"]
        idx = segment_frames(seg, trace)
        labels = [labels_by_clip[seg["clip"]][i] for i in idx]
        votes = Counter(l for l in labels if l not in IGNORED)
        pred = votes.most_common(1)[0][0] if votes else NO_PERSON
        out.append({"pred": pred, "frames": len(idx),
                    "frame_acc": sum(l == seg["label"] for l in labels) / max(len(labels), 1)})
    return out


def score(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    labels = [c for c in CLASSES if c in set(y_true)]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0, output_dict=True),
        "report_text": classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0),
    }


def diagnostics(seg: dict, trace: list[dict], fps_source: float) -> dict:
    """Measured signals per segment, used to explain failures."""
    idx = segment_frames(seg, trace) or list(range(len(trace)))
    frames = [trace[i] for i in idx]
    pose = [f["pose"] for f in frames if f["pose"]]
    heights = [p["body_h"] for p in pose]
    cam = [f["cam_offset_x"] for f in frames]
    dur = len(frames) / max(fps_source, 1)
    return {
        "duration_s": round(dur, 2),
        "pose_found": round(len(pose) / max(len(frames), 1), 3),
        "legs_visible": round(sum(p["legs_visible"] for p in pose) / max(len(pose), 1), 3),
        "body_height_px": round(float(np.median(heights)), 1) if heights else 0.0,
        "camera_px_per_s": round(abs(cam[-1] - cam[0]) / max(dur, 1e-3), 1) if len(cam) > 1 else 0.0,
        "people": max((f["n_people"] for f in frames), default=0),
    }


def speed(traces: dict) -> dict:
    stages = ["preprocess", "infer", "postprocess", "render"]
    rows = [t for d in traces.values() for t in d["timings"][WARMUP_FRAMES:]]
    if not rows:
        rows = [t for d in traces.values() for t in d["timings"]]
    ms = {s: float(np.median([r[s] for r in rows])) for s in stages}
    total = float(np.median([sum(r[s] for s in stages) for r in rows]))
    return {"ms_per_frame": round(total, 1), "fps": round(1000 / total, 1),
            "stage_ms": {s: round(v, 1) for s, v in ms.items()}, "frames_timed": len(rows),
            "yolo": all(d.get("yolo") for d in traces.values())}


def machine_name() -> str:
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10).stdout
        name = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"][0]
    except Exception:
        name = platform.processor() or platform.machine()
    return f"{name}, {platform.system()} {platform.release()}, CPU only"


def save_confusion(y_true, y_pred, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    labels = [c for c in CLASSES + [NO_PERSON] if c in set(y_true) | set(y_pred)]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=120)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, xticks_rotation=45, ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main
def evaluate(split: str, rerun: bool = False, use_yolo: bool = True, write: bool = True):
    segments = load_segments(split)
    if not segments:
        sys.exit(f"No labelled clips in data/{split}. Run scripts/make_splits.py first.")
    clips = sorted({s["clip"] for s in segments})
    print(f"{split}: {len(segments)} segments in {len(clips)} videos", flush=True)
    traces = get_traces(split, clips, rerun, use_yolo)

    tuned = Params.load()
    params = {"baseline": tuned, "main_m2": Params(), "main": tuned}
    y_true = [s["label"] for s in segments]
    results = {}
    for m, p in params.items():
        preds = predict_segments(segments, traces, p, m)
        results[m] = {"preds": preds, **score(y_true, [x["pred"] for x in preds])}
    spd = speed(traces)
    if not write:
        return segments, traces, results

    # Test log: one row per segment.
    METRICS.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for i, seg in enumerate(segments):
        d = traces[seg["clip"]]
        got = results["main"]["preds"][i]["pred"]
        log_rows.append({
            "id": f"{'T' if split == 'test' else 'V'}{i + 1:03d}",
            "input": f"{seg['clip'].parent.name}/{seg['clip'].name}"
                     + ("" if seg["end"] == float("inf") else f" [{seg['start']:g}-{seg['end']:g}s]"),
            "condition": seg["condition"], "expected": seg["label"], "got": got,
            "ok": "✔" if got == seg["label"] else "✘",
            "baseline_got": results["baseline"]["preds"][i]["pred"],
            "m2_got": results["main_m2"]["preds"][i]["pred"],
            "frame_acc": round(results["main"]["preds"][i]["frame_acc"], 3),
            **diagnostics(seg, d["trace"], d["fps_source"]),
        })
    log_path = METRICS / f"test_log_{split}.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)

    for m in METHODS:
        save_confusion(y_true, [x["pred"] for x in results[m]["preds"]],
                       f"{METHODS[m]} - {split}", FIGURES / f"confusion_{split}_{m}.png")

    by_class = Counter(y_true)
    by_dataset = Counter("gmdcsa24" if s["clip"].name.startswith("gmdcsa") else
                         "hmdb51" if s["clip"].name.startswith("hmdb") else "own" for s in segments)
    summary = {
        "split": split, "date": str(date.today()), "segments": len(segments), "videos": len(clips),
        "per_class": dict(by_class), "per_dataset": dict(by_dataset), "machine": machine_name(), "speed": spd,
        "params_file": "configs/tuned_params.json" if (ROOT / "configs" / "tuned_params.json").exists() else "defaults",
        "methods": {m: {"name": METHODS[m], "accuracy": round(r["accuracy"], 4), "macro_f1": round(r["macro_f1"], 4),
                        "per_class": {c: {k: round(v, 3) for k, v in r["report"][c].items()}
                                      for c in r["report"] if c in CLASSES}}
                    for m, r in results.items()},
    }
    (METRICS / f"eval_{split}.json").write_text(json.dumps(summary, indent=2))
    write_markdown(split, summary, results, log_rows)

    print(f"\n{split}: {len(segments)} segments  " + "  ".join(f"{dict(by_class)}".split()))
    print(f"{'method':52s} {'accuracy':>9s} {'macro-F1':>9s}")
    for m, r in results.items():
        print(f"{METHODS[m]:52s} {r['accuracy']:9.3f} {r['macro_f1']:9.3f}")
    print(f"speed: {spd['ms_per_frame']} ms/frame = {spd['fps']} FPS on {summary['machine']} "
          f"({'YOLO on' if spd['yolo'] else 'no YOLO'}, after {WARMUP_FRAMES}-frame warm-up)")
    print(f"\nMain method, per class:\n{results['main']['report_text']}")
    print(f"saved {log_path}, {METRICS / f'eval_{split}.json'}, docs/results_{split}.md")
    return segments, traces, results


def write_markdown(split, summary, results, log_rows):
    s = summary
    lines = [f"# Results on the {split} split", "",
             f"Generated by `python src/evaluate.py --split {split}` on {s['date']}. "
             f"{s['segments']} labelled segments from {s['videos']} videos "
             f"({', '.join(f'{k}: {v}' for k, v in s['per_dataset'].items())}). "
             f"Per class: {', '.join(f'{c} {s['per_class'].get(c, 0)}' for c in CLASSES)}.", "",
             "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for m, r in s["methods"].items():
        lines.append(f"| {r['name']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    sp = s["speed"]
    lines += ["", f"**Speed:** {sp['ms_per_frame']} ms per frame = **{sp['fps']} FPS** (median over "
                  f"{sp['frames_timed']} frames after a {WARMUP_FRAMES}-frame warm-up per clip, "
                  f"{'YOLO on' if sp['yolo'] else 'YOLO off'}) on {s['machine']}. "
                  "Per stage: " + ", ".join(f"{k} {v} ms" for k, v in sp["stage_ms"].items()) + ".",
              "", "## Main method, per class", "", "| Class | Precision | Recall | F1 | Support |",
              "|---|---:|---:|---:|---:|"]
    for c, v in s["methods"]["main"]["per_class"].items():
        lines.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |")
    lines += ["", "## Confusion matrices", ""]
    for m in METHODS:
        lines.append(f"![{METHODS[m]}](figures/confusion_{split}_{m}.png)")
    fails = [r for r in log_rows if r["ok"] == "✘"]
    lines += ["", f"## Test log ({len(log_rows) - len(fails)} ✔ / {len(fails)} ✘)", "",
              f"Full log with diagnostics: `outputs/metrics/test_log_{split}.csv`.", "",
              "| ID | Input | Condition | Expected | Got | | Baseline |", "|---|---|---|---|---|---|---|"]
    for r in log_rows:
        cond = r["condition"][:70] + ("…" if len(r["condition"]) > 70 else "")
        lines.append(f"| {r['id']} | {r['input']} | {cond} | {r['expected']} | {r['got']} | {r['ok']} | {r['baseline_got']} |")
    (ROOT / "docs" / f"results_{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--rerun", action="store_true", help="ignore cached runs and process every clip again")
    ap.add_argument("--no-yolo", action="store_true")
    args = ap.parse_args()
    evaluate(args.split, args.rerun, not args.no_yolo)


if __name__ == "__main__":
    main()
