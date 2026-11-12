"""Tune the rule thresholds on the VAL split only, and save them to
configs/tuned_params.json (which run.py and evaluate.py then use).

    python scripts/tune.py               # ~2-5 minutes after the val clips are cached
    python scripts/tune.py --trials 3000

Random search: each trial draws a full parameter set, replays the cached val
features through the rules (no neural networks are re-run), and scores
macro-F1 over the val segments. The baseline and the main method are tuned
separately, with the same number of trials, so the comparison is fair. The
test split is never read here.
"""
import argparse
import random
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import evaluate, predict_segments, score  # noqa: E402
from src.params import Params  # noqa: E402

MAIN_SPACE = {
    "sit_knee_angle_max": (100, 160),
    "sit_hip_angle_max": (100, 160),
    "sit_thigh_ratio_max": (0.2, 0.7),
    "lying_torso_angle_min": (40, 80),
    "walk_hip_speed_min": (0.05, 0.6),
    "walk_leg_speed_min": (0.2, 1.5),
    "run_hip_speed_min": (0.6, 2.5),
    "run_leg_speed_min": (1.0, 4.0),
    "camera_compensation": [True, False],
    "seated_first": [True, False],
    "require_person_box": [True, False],
    "walk_stride_min": (0.0, 0.8),
    "smoothing_window": [1, 5, 9, 15, 21],
    "persist_frames": [1, 3, 5, 8],
    "sit_persist_frames": [1, 5, 10, 20, 30],
}
BASE_SPACE = {
    "base_upright_aspect_min": (1.0, 2.5),
    "base_lying_aspect_max": (0.4, 1.2),
    "base_walk_speed_min": (0.05, 0.8),
    "base_run_speed_min": (0.6, 3.0),
    "base_smoothing_window": [1, 5, 9, 15, 21],
    "base_persist_frames": [1, 3, 5, 8],
}


def sample(space: dict, rng: random.Random, base: Params) -> Params:
    vals = {}
    for k, v in space.items():
        if isinstance(v, list):
            vals[k] = rng.choice(v)
        else:
            lo, hi = v
            vals[k] = round(rng.uniform(lo, hi), 3)
    return replace(base, **vals)


def dataset_of(seg) -> str:
    return seg["clip"].name.split("-")[0]


def objective_fn(segments, traces, y_true, method):
    """Mean of the macro-F1 on each dataset (room footage and movie clips count
    equally, however many segments each has), then overall macro-F1 to break
    ties. Milestone 3 used overall macro-F1 alone, which let 55 movie walkers
    outvote 15 room sitters (see docs/error_analysis.md)."""
    groups = {}
    for i, s in enumerate(segments):
        groups.setdefault(dataset_of(s), []).append(i)

    def objective(p):
        preds = [x["pred"] for x in predict_segments(segments, traces, p, method)]
        per_ds = [score([y_true[i] for i in idx], [preds[i] for i in idx])["macro_f1"] for idx in groups.values()]
        overall = score(y_true, preds)
        return (sum(per_ds) / len(per_ds), overall["macro_f1"], overall["accuracy"])
    return objective


def search(space, method, segments, traces, y_true, trials, rng, start: Params):
    objective = objective_fn(segments, traces, y_true, method)

    best_p, best = start, objective(start)
    for i in range(trials):
        p = sample(space, rng, best_p)
        s = objective(p)
        if s > best:
            best_p, best = p, s
            print(f"  [{method}] trial {i + 1}: dataset-balanced macro-F1 {s[0]:.3f}, "
                  f"overall macro-F1 {s[1]:.3f}, accuracy {s[2]:.3f}", flush=True)
    return best_p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=106)
    args = ap.parse_args()

    print("Loading val features (runs the pipeline on uncached clips)...")
    segments, traces, _ = evaluate("val", write=False)
    y_true = [s["label"] for s in segments]
    rng = random.Random(args.seed)

    # Start from the current tuned file (if any), so a new search only replaces
    # it with something that scores HIGHER on val. Random search over ~20
    # parameters is noisy: a re-run can land on a worse set than one we already have.
    obj = objective_fn(segments, traces, y_true, "main")
    print(f"M2 hand-set values on val: dataset-balanced macro-F1 {obj(Params())[0]:.3f}")
    start = Params.load()
    print(f"current configs/tuned_params.json on val: {obj(start)[0]:.3f} (kept unless beaten)")
    main_p, main_s = search(MAIN_SPACE, "main", segments, traces, y_true, args.trials, rng, start)
    base_p, base_s = search(BASE_SPACE, "baseline", segments, traces, y_true, args.trials, rng, start)

    # One file holds both: main-method fields from main_p, baseline fields from base_p.
    merged = replace(main_p, **{k: getattr(base_p, k) for k in BASE_SPACE})
    merged.save(tuned_on="val", date=str(date.today()), trials=args.trials, seed=args.seed,
                objective="mean of per-dataset macro-F1", val_segments=len(segments),
                val_balanced_macro_f1={"main": round(main_s[0], 4), "baseline": round(base_s[0], 4)},
                val_macro_f1={"main": round(main_s[1], 4), "baseline": round(base_s[1], 4)},
                val_accuracy={"main": round(main_s[2], 4), "baseline": round(base_s[2], 4)})
    print(f"\nval dataset-balanced macro-F1: main {main_s[0]:.3f}, baseline {base_s[0]:.3f}")
    print("saved configs/tuned_params.json:")
    for k, v in asdict(merged).items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
