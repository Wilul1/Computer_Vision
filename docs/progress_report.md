# Milestone 2: 50% Progress Report

**Project:** Real-Time and Video-Based Motion Detection and Activity Monitoring System
**Tag:** `cp2-50` · **Date:** TODO

> The owners below are a suggested split. Change them to match who actually did each task, because each
> member must explain the file they own during the code review.

| Task (from proposal) | Owner | Status | Evidence |
|---|---|---|---|
| Python environment, pinned requirements, README setup | Wilmark | Done | `requirements.txt`, `README.md` |
| Webcam and video input module | Wilmark | Done | `src/predict.py` (camera index or file) |
| Preprocessing: resize, CLAHE, blur | Cindy | Done | `src/preprocess.py`, `docs/figures/preprocess_*.jpg` |
| Motion baseline: MOG2 + contours, event counter | Cindy | Done (code) | `src/motion.py`, `scripts/eval_motion_baseline.py` |
| Person detection: pretrained YOLO11n | Zheniel | Done | `src/detector.py` |
| Pose estimation: MediaPipe 33 landmarks | Zheniel | Done | `src/pose.py` |
| Features: joint angles, thigh ratio, torso tilt, speeds | Wilmark | Done | `src/features.py` |
| Rule-based classifier + temporal smoothing | Wilmark | Done (first thresholds) | `src/classifier.py`, `src/config.py` |
| Overlay UI: boxes, skeleton, label, FPS | Cindy | Done | `src/visualize.py` |
| Record dataset, split by source, data card | All | TODO | `data/README.md`, `data/splits.csv` |
| Baseline numbers on val | All | TODO | `outputs/metrics/activity_val.json`, `motion_baseline.json` |
| Threshold tuning on val, fall detection, evaluation in different lighting | All | Not started (Phase 3–4) | |

## Baseline numbers (validation split)

| Metric | Value | Source |
|---|---|---|
| Activity accuracy (clip level) | TODO | `outputs/metrics/activity_val.json` |
| Activity macro-F1 | TODO | same |
| MOG2 motion events vs hand count (MAE, 2–3 clips) | TODO | `outputs/metrics/motion_baseline.json` |
| Processing speed, CPU (FPS) | TODO | same |

## Percent complete

Weighted by effort over the planned tasks for the whole project: TODO. Count the done tasks above and
explain the weights.

## Blockers
- The dataset still has to be recorded. Each member records at least 2 sessions in different rooms and lighting.

## Next dates
- TODO: dataset recorded and split
- Week 4–5 (Oct 31 – Nov 12): tune thresholds on val, add fall detection, test lighting conditions (80% check)
- Week 6 (Nov 13–20): final testing, test-split evaluation, presentation and live demo

## Correction to the proposal timeline
Weeks 3–5 of the proposal mention squats, push-ups, lunges, jumping jacks and repetition counting. Those were
carried over by mistake. The project classifies standing, sitting, walking, running and lying, as stated in the
objectives.
