"""Run the test cases in tests/test_cases.csv and record the results.

    python scripts/run_test_cases.py                 # every file-based case
    python scripts/run_test_cases.py --webcam 0      # also the live webcam cases (10 s each)

For each case: the label shown most often over the clip (or the photo's label),
its confidence (share of frames that agree), speed, and pass/fail against the
expected label written BEFORE the run. Annotated outputs go to
outputs/test_cases/, the table to docs/test_cases.md.

A case expecting "no person" or "unknown" passes when that is what the system
reported for most frames; otherwise frames with no person / hidden legs do not
vote, as in evaluate.py.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifier import IGNORED  # noqa: E402
from src.params import Params  # noqa: E402
from src.pipeline import MonitorPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "test_cases.csv"
OUT = ROOT / "outputs" / "test_cases"
DOC = ROOT / "docs" / "test_cases.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def verdict(shown: list[str], raw: list[str]) -> tuple[str, float]:
    """Label for the whole input and the share of frames that agree with it."""
    ignored = Counter(r for r in raw if r in IGNORED)
    if ignored and sum(ignored.values()) >= len(raw) / 2:
        label, n = ignored.most_common(1)[0]
        return label, n / len(raw)
    votes = Counter(s for s in shown if s not in IGNORED)
    if not votes:
        return "no person", 1.0
    label, n = votes.most_common(1)[0]
    return label, n / sum(votes.values())


def run_case(pipe: MonitorPipeline, source, out_path: Path, seconds: float | None = None) -> dict:
    pipe.reset()
    shown, raw, times = [], [], []
    if isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(source))
        t0 = time.perf_counter()
        r = pipe.process(frame, 0.0)
        times.append((time.perf_counter() - t0) * 1000)
        cv2.imwrite(str(out_path.with_suffix(".jpg")), r["annotated"])
        shown, raw = [r["activity"]], [r["raw_activity"]]
    else:
        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer, i, start = None, 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter() - start if isinstance(source, int) else i / fps
            if seconds and t > seconds:
                break
            t0 = time.perf_counter()
            r = pipe.process(frame, t)
            times.append((time.perf_counter() - t0) * 1000)
            if writer is None:
                h, w = r["annotated"].shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps if not isinstance(source, int) else 15, (w, h))
            writer.write(r["annotated"])
            if isinstance(source, int):
                cv2.imshow("test case (q to stop)", r["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            shown.append(r["activity"])
            raw.append(r["raw_activity"])
            i += 1
        cap.release()
        if writer:
            writer.release()
        if isinstance(source, int):
            cv2.destroyAllWindows()
    label, conf = verdict(shown, raw)
    warm = times[10:] or times
    ms = sorted(warm)[len(warm) // 2]
    return {"actual": label, "confidence": conf, "frames": len(shown), "ms": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", type=int, default=None, help="also run the live cases with this camera")
    ap.add_argument("--seconds", type=float, default=10, help="length of each live case")
    args = ap.parse_args()

    with open(CASES, newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = MonitorPipeline(Params.load())
    rows = []
    try:
        for c in cases:
            live = c["input"] == "webcam"
            if live and args.webcam is None:
                rows.append({**c, "actual": "(run live: --webcam 0)", "confidence": "", "ms": "", "pass": "–"})
                continue
            if live:
                input(f"\n{c['id']}: {c['condition']}\nPress Enter to start ({args.seconds:.0f} s)...")
                res = run_case(pipe, args.webcam, OUT / f"{c['id']}_webcam.mp4", args.seconds)
            else:
                src = ROOT / "tests" / c["input"]
                res = run_case(pipe, src, OUT / f"{c['id']}_{src.stem}.mp4")
            ok = res["actual"] == c["expected"]
            rows.append({**c, "actual": res["actual"], "confidence": f"{res['confidence']:.2f}",
                         "ms": f"{res['ms']:.0f}", "pass": "✔" if ok else "✘"})
            print(f"{c['id']} expected {c['expected']:9s} got {res['actual']:9s} "
                  f"({res['confidence']:.2f}, {res['ms']:.0f} ms/frame) {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        pipe.close()

    ran = [r for r in rows if r["pass"] in ("✔", "✘")]
    hard = [r for r in ran if r["hard"] == "yes"]
    lines = ["# Test cases", "",
             "Written before the run (inputs, conditions and expected labels in `tests/test_cases.csv`), then run "
             f"with `python scripts/run_test_cases.py` on {date.today()}. Annotated outputs: `outputs/test_cases/`.", "",
             f"**{sum(r['pass'] == '✔' for r in ran)} of {len(ran)} passed** "
             f"({sum(r['pass'] == '✔' for r in hard)} of {len(hard)} hard cases). "
             "Confidence = share of frames that agree with the reported label. "
             "Speed = median ms per frame after 10 warm-up frames.", "",
             "| ID | Input | Condition | Expected | Actual | Conf. | ms/frame | Pass |",
             "|---|---|---|---|---|---:|---:|:-:|"]
    for r in rows:
        lines.append(f"| {r['id']} | `{r['input']}` | {r['condition']} | {r['expected']} | {r['actual']} | "
                     f"{r['confidence']} | {r['ms']} | {r['pass']} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {DOC}")


if __name__ == "__main__":
    main()
