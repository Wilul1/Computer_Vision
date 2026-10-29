"""Reproduce every validation number in one command (for the in-class check).

    python scripts/run_all_eval.py              # val split, YOLO on
    python scripts/run_all_eval.py --no-yolo    # faster

Runs the activity evaluation and the MOG2 motion baseline, then writes
outputs/metrics/summary.md with the table for the progress report.
"""
import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics"


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-yolo", action="store_true")
    args = p.parse_args()

    activity_ok = run("eval_activity.py", "--split", "val", *(["--no-yolo"] if args.no_yolo else [])) == 0
    motion_ok = run("eval_motion_baseline.py") == 0

    lines = [f"# Validation results ({date.today()})", "",
             "Reproduce with `python scripts/run_all_eval.py`. Split: **val** (test is locked).", "",
             "| Method | Metric | Value |", "|---|---|---|"]
    act_file, mot_file = METRICS / "activity_val.json", METRICS / "motion_baseline.json"
    if activity_ok and act_file.exists():
        a = json.loads(act_file.read_text())
        lines += [f"| Pose + rules (main) | clip accuracy ({a['clips']} clips) | {a['accuracy']:.3f} |",
                  f"| Pose + rules (main) | macro-F1 | {a['macro_f1']:.3f} |",
                  f"| Pose + rules (main) | FPS, CPU{' (no YOLO)' if args.no_yolo else ''} | {a['mean_fps']:.1f} |"]
    else:
        lines.append("| Pose + rules (main) | – | not run (no val clips?) |")
    if motion_ok and mot_file.exists():
        m = json.loads(mot_file.read_text())
        lines += [f"| MOG2 + contours (baseline) | event-count MAE vs hand count ({m['clips']} clips) | {m['mean_abs_error']:.2f} |",
                  f"| MOG2 + contours (baseline) | count accuracy | {m['count_accuracy']:.1%} |",
                  f"| MOG2 + contours (baseline) | FPS, CPU | {m['mean_fps']:.1f} |"]
    else:
        lines.append("| MOG2 + contours (baseline) | – | not run (fill data/annotations/motion_counts.csv) |")
    if activity_ok and act_file.exists():
        conf = a["confusion"]
        cols = sorted({c for row in conf.values() for c in row})
        lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
                  "| true \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {t} | " + " | ".join(str(conf[t].get(c, 0)) for c in cols) + " |" for t in conf]

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved {METRICS / 'summary.md'}")
    sys.exit(0 if activity_ok and motion_ok else 1)


if __name__ == "__main__":
    main()
