# Real-Time and Video-Based Motion Detection and Activity Monitoring

CSST 106 Computer Vision, final project. Laguna State Polytechnic University.
Alabat Cindy I · Porcopio Zheniel Ivan · Veridiano Wilmark

The system detects people and classifies **standing, sitting, walking, running and lying** from a laptop
webcam, phone camera or recorded video. It uses pretrained models and rule-based geometry, with no custom
training:

```
frame ─► preprocess ─┬─► MOG2 background subtraction ─► motion boxes / motion events
 (resize 640,        ├─► YOLO11n ─────────────────────► person boxes (primary = largest)
  CLAHE, blur)       └─► MediaPipe Pose (33 landmarks) ─► joint angles, thigh ratio, torso tilt,
                                                          hip & leg speed (body-heights/s)
                                                          ─► rules + 9-frame majority vote
                                                          ─► label + skeleton + boxes + FPS overlay
```

## Setup (CPU only, Python 3.10–3.13)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads two model files into `models/`: the MediaPipe pose model (~6 MB) and YOLO11n (~6 MB).

## Run the prototype

```bash
python src/predict.py --input data/test/walking/<clip>.mp4 --output outputs/
python src/predict.py --input path/to/photo.jpg --output outputs/
python src/predict.py --input 0                 # live webcam, press q to stop
```

Each run writes `outputs/<name>_annotated.mp4` (or `.jpg`) and `outputs/<name>_log.csv`, a per-frame log with the
activity, angles, speeds and FPS. Options: `--show` (live window for files), `--no-yolo` (faster), `--no-enhance`
(skip CLAHE), `--max-seconds N`.

## Dataset

The videos are not in git. Download them from **Google Drive: `<PASTE LINK>`** and unzip into `data/raw/`, or
record your own:

```bash
python scripts/record_clips.py --activity walking --source s01-wilmark-room-webcam --seconds 10 --count 3
python scripts/make_splits.py        # 70/15/15 split BY SOURCE -> data/train|val|test, data/splits.csv
```

See [data/README.md](data/README.md) for the data card.

## Evaluate

```bash
python scripts/eval_activity.py --split val          # accuracy, macro-F1, confusion matrix, FPS
python scripts/eval_motion_baseline.py               # MOG2 events vs hand count, FPS
python scripts/preprocess_strip.py --input <clip>    # before/after figure in docs/figures/
```

Results are saved to `outputs/metrics/*.json`. The test split is only used for the final evaluation.

## Layout

```
src/            config.py (all thresholds) · preprocess.py · motion.py · detector.py · pose.py
                features.py · classifier.py · visualize.py · pipeline.py · predict.py
scripts/        record_clips.py · make_splits.py · eval_activity.py · eval_motion_baseline.py · preprocess_strip.py
data/           raw/<activity>/<source>__NN.mp4 · train|val|test/<activity>/ · splits.csv · annotations/
docs/           progress_report.md · figures/
outputs/        annotated results and metrics
```
