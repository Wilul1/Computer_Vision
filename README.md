# Real-Time and Video-Based Motion Detection and Activity Monitoring

CSST 106 Computer Vision, final project. Laguna State Polytechnic University.
Alabat Cindy I · Porcopio Zheniel Ivan · Veridiano Wilmark

The system detects people and classifies **standing, sitting, walking, running and lying** from a laptop
webcam, phone camera or recorded video. It uses pretrained models and rule-based geometry, with no custom
training. The rule thresholds are tuned on the validation split.

```
frame ─► preprocess() ─► infer() ─────────────────────────────► postprocess() ─────────────────► render()
         resize ≤640     ├ MOG2 background subtraction ──► blob shape + speed ─► BASELINE rules ┐  boxes, skeleton,
         CLAHE (Lab L)   ├ YOLO11n person boxes                                                  ├► label, FPS,
         gray + blur     ├ MediaPipe Pose, 33 landmarks ─► angles, thigh ratio, torso tilt,      │  CSV row
                         └ LK optical flow (background) ─► hip speed minus camera motion, leg    │
                                                           swing ─► MAIN rules ─► smoothing +   ─┘
                                                           persistence
```

## Setup (CPU only, Python 3.10–3.13)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads two model files into `models/`: the MediaPipe pose model (~6 MB) and YOLO11n (~6 MB).

## The app

```bash
python app.py                     # then open http://127.0.0.1:7860
```

A browser interface with three tabs: **Video** (upload, get the annotated video, the main activity, confidence,
time per activity, a timeline, motion events and ms per frame), **Photo**, and **Webcam (live)**. The test-case
clips are built in as examples. Models are loaded once at start-up. For a desktop window instead of the browser:
`python src/run.py --webcam 0`.

## Run: one command, raw input to visible output

```bash
python src/run.py --input data/test/ --output outputs/run_2026-11-12/   # every image/video in a folder
python src/run.py --input demo.mp4 --output outputs/demo_annotated.mp4    # one video
python src/run.py --input photo.jpg --output outputs/                      # one image
python src/run.py --webcam 0                                               # live webcam, press q to stop
```

Each input gives an annotated video or image plus `<name>_log.csv`, a per-frame log with the activity, the
baseline's activity, angles, speeds and FPS. A folder run also writes `summary.csv` with one row per input.
Options: `--method baseline` (show the baseline's label), `--show`, `--no-yolo` (faster), `--max-seconds N`,
`--params <file>`. Without `--params`, the values tuned on val are used (`configs/tuned_params.json`).

## Dataset

The videos are not in git. **You don't need them to reproduce the reported numbers** (see below). To rebuild them
(≈3.2 GB of downloads) in one command:

```bash
python scripts/get_data.py --root D:/datasets
```

which does the following:

```bash
# HMDB51: https://huggingface.co/datasets/jili5044/hmdb51 -> unzip, keep walk/ and run/
python scripts/import_public.py hmdb51 D:/datasets/hmdb51/hmdb51 --limit-per-source 3
# GMDCSA24: git clone https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
python scripts/import_public.py gmdcsa24 D:/datasets/GMDCSA24
python scripts/make_splits.py        # split BY SOURCE (movie / subject) -> data/train|val|test, data/splits.csv
```

`data/splits.csv` is committed, so the split is the same on every machine. Own clips can be added with
`scripts/record_clips.py`; new sources join the split without moving existing ones. See
[data/README.md](data/README.md) for the data card.

## Reproduce the reported numbers

```bash
python src/evaluate.py --split test         # prints the table in docs/results_test.md and the report
```

This works on a fresh clone **without the videos**: the per-frame features of every val and test video are
committed in `outputs/cache/` (≈14 MB), and `evaluate.py` scores them with the committed parameter files
(`configs/tuned_params.json`, and `configs/tuned_params_m3.json` for the Milestone 3 row). Accuracy, F1, the
per-condition table and the speed (measured on the machine named in the output) all come out identical. With the
videos present, `--rerun` re-runs the whole pipeline instead. Tuning is seeded (`--seed 106`).

## Test cases

```bash
python scripts/run_test_cases.py             # 13 file-based cases -> docs/test_cases.md
python scripts/run_test_cases.py --webcam 0  # plus the 2 live webcam cases
```

The cases, with their expected labels written before running, are in `tests/test_cases.csv`, and the clips in
`tests/cases/` (cut from MIT-licensed GMDCSA24 by `scripts/make_test_cases.py`). 7 of them are hard cases taken
from the Milestone 3 error analysis.

## Evaluate, tune, analyse

```bash
python src/evaluate.py --split test         # baseline vs main on the held-out test set, per-class report,
                                            # confusion matrices, speed, test log -> docs/results_test.md
python scripts/tune.py                      # tunes thresholds on VAL only -> configs/tuned_params.json
python scripts/failure_gallery.py           # failure frames + suggested causes -> docs/error_analysis.md
python scripts/preprocess_strip.py --input data/val   # before/after preprocessing figure
python scripts/eval_motion_baseline.py      # MOG2 motion events vs hand count (static-camera clips)
```

The first `evaluate.py` run processes every clip through the full pipeline and caches the per-frame features in
`outputs/cache/`; later runs reuse them (`--rerun` to start over). The three methods are scored on exactly the same
frames. Results: [docs/results_test.md](docs/results_test.md), error analysis and improvement plan:
[docs/error_analysis.md](docs/error_analysis.md), reasons for each setting:
[docs/design_choices.md](docs/design_choices.md).

## Layout

```
src/        run.py (entry point) · pipeline.py (4 stages) · preprocess.py · motion.py · detector.py · pose.py
            camera_motion.py · features.py · classifier.py · params.py · visualize.py · evaluate.py · config.py
app.py      the browser app (Gradio)
scripts/    get_data.py · import_public.py · make_splits.py · tune.py · failure_gallery.py · record_clips.py
            run_test_cases.py · make_test_cases.py · preprocess_strip.py · eval_motion_baseline.py · make_report_pdf.py
configs/    tuned_params.json (tuned on val) · tuned_params_m3.json (Milestone 3 values, for comparison)
tests/      test_cases.csv · cases/ (test-case clips and photos)
data/       raw/ · train|val|test/ · splits.csv · annotations/ (segments.csv, clip_info.csv, motion_counts.csv)
docs/       results_test.md · error_analysis.md · design_choices.md · progress_report.md · figures/ · failures/
outputs/    annotated results, metrics, cache
```
