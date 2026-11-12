"""Failure gallery for the error analysis: one annotated frame per failed test
segment, a SUGGESTED cause from measured signals, and counts per cause.

    python scripts/failure_gallery.py            # after: python src/evaluate.py --split test

The suggested cause comes from the diagnostics in outputs/metrics/test_log_test.csv
and the clip's recorded condition, checked in this order (the first match wins):
  Person not found          pose found in < 50% of the segment's frames
  Cut off by the frame      legs hidden in > 50% of the frames with a pose
  Transition / short        segment < 1.5 s (smoothing and the 0.5 s speed window dominate)
  Limb motion in place      sitting/standing called walking/running with a static camera
  Moving camera             condition says moving camera, or the background moves > 40 px/s
  Look-alike posture        sitting called standing
  Far away / low res        median body height < 90 px
  Night / low light         night or lights off AND pose found in < 90% of frames
  Speed near a threshold    none of the above (walking vs running vs standing by speed)
Each suggestion is a starting point: LOOK at the frame and correct the cause
column in docs/error_analysis.md if it is wrong.

Writes docs/failures/<ID>.jpg and the gallery + counts table in
docs/error_analysis.md (between the GALLERY markers, so text you write around
it is kept).
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.params import Params  # noqa: E402
from src.pipeline import MonitorPipeline  # noqa: E402

ROOT = config.ROOT
LOG = ROOT / "outputs" / "metrics" / "test_log_test.csv"
OUT_DIR = ROOT / "docs" / "failures"
DOC = ROOT / "docs" / "error_analysis.md"
START, END = "<!-- GALLERY START -->", "<!-- GALLERY END -->"


def suggest_cause(r: dict) -> str:
    cond = r["condition"].lower()
    moving_cam = "moving camera" in cond or float(r["camera_px_per_s"]) > 40
    if float(r["pose_found"]) < 0.5:
        return "Person not found"
    if float(r["legs_visible"]) < 0.5:
        return "Cut off by the frame"
    if float(r["duration_s"]) < 1.5:
        return "Transition / short segment"
    if r["expected"] in ("sitting", "standing") and r["got"] in ("walking", "running") and not moving_cam:
        return "Limb motion in place read as walking"
    if moving_cam:
        return "Moving camera"
    if r["expected"] == "sitting" and r["got"] == "standing":
        return "Look-alike posture"
    if 0 < float(r["body_height_px"]) < 90:
        return "Far away / low resolution"
    if ("night" in cond or "lights off" in cond) and float(r["pose_found"]) < 0.9:
        return "Night / low light"
    return "Speed near a threshold"


def grab_frame(r: dict) -> "cv2.Mat | None":
    """Re-run the pipeline up to the middle of the segment and return that annotated frame."""
    name = r["input"].split(" [")[0]
    clip = next((ROOT / "data" / "test").rglob(Path(name).name), None)
    if clip is None:
        return None
    start, end = 0.0, None
    if " [" in r["input"]:
        start, end = (float(x) for x in r["input"].split(" [")[1].rstrip("s]").split("-"))
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = int(((start + end) / 2 if end else n / fps / 2) * fps)
    pipe = MonitorPipeline(Params.load(), use_yolo=True)
    out, i = None, 0
    while i <= target:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipe.process(frame, i / fps)["annotated"]
        i += 1
    cap.release()
    pipe.close()
    return out


def main():
    import argparse
    global OUT_DIR, DOC
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="folder for the failure frames")
    ap.add_argument("--doc", default=str(DOC), help="markdown file that gets the gallery block")
    args = ap.parse_args()
    OUT_DIR, DOC = Path(args.out_dir), Path(args.doc)
    if not LOG.exists():
        sys.exit("Run python src/evaluate.py --split test first.")
    with open(LOG, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fails = [r for r in rows if r["ok"] != "✔"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):
        old.unlink()

    lines, causes = [], Counter()
    for i, r in enumerate(fails, 1):
        cause = suggest_cause(r)
        causes[cause] += 1
        img = grab_frame(r)
        if img is not None:
            h, w = img.shape[:2]
            img = cv2.resize(img, (360, int(h * 360 / w)))
            cv2.imwrite(str(OUT_DIR / f"{r['id']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[{i}/{len(fails)}] {r['id']} {r['expected']} -> {r['got']}: {cause}", flush=True)
        lines.append(f"| {r['id']} | ![{r['id']}]({OUT_DIR.name}/{r['id']}.jpg) | {r['input']} | {r['expected']} | "
                     f"{r['got']} | {cause} | pose {float(r['pose_found']):.0%}, legs {float(r['legs_visible']):.0%}, "
                     f"{float(r['body_height_px']):.0f} px tall, camera {float(r['camera_px_per_s']):.0f} px/s, "
                     f"{float(r['duration_s']):.1f} s | {r['condition'][:60]} |")

    pairs = Counter((r["expected"], r["got"]) for r in fails)
    block = [START, "",
             f"**{len(fails)} failures out of {len(rows)} test segments** (main method). Causes are suggested by "
             "`scripts/failure_gallery.py` from measured signals, then checked by looking at each frame.", "",
             "| Cause | Count |", "|---|---:|"]
    block += [f"| {c} | {n} |" for c, n in causes.most_common()]
    block += ["", "Most common confusions (expected → got): "
              + ", ".join(f"{e} → {g} ({n})" for (e, g), n in pairs.most_common(6)) + ".", "",
              "| ID | Frame | Input | Expected | Got | Cause | Measured | Condition |",
              "|---|---|---|---|---|---|---|---|", *lines, "", END]
    text = DOC.read_text(encoding="utf-8") if DOC.exists() else "# Error analysis\n\n"
    if START in text and END in text:
        text = text[:text.index(START)] + "\n".join(block) + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    DOC.write_text(text, encoding="utf-8")
    print(f"\n{dict(causes.most_common())}\nsaved {DOC} and {len(fails)} frames in {OUT_DIR}")


if __name__ == "__main__":
    main()
