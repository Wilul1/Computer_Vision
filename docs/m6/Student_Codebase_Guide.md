# Student Codebase Guide — What Each File Does

**Project:** MotionSense / Activity Monitor  
**How to use this:** Practice saying each “What I’d say” out loud. If a reviewer opens a file, jump to that section.

**One-sentence map of the whole repo**

> `app.py` + `ui.py` are the browser demo. Everything important for classification lives in `src/` — especially `pipeline.py`, which calls preprocess → detect → pose → features → rules → draw. `scripts/` builds data and slides. `configs/tuned_params.json` holds the thresholds we tuned on validation.

---

# Big picture (say this first)

“Our code is a **pipeline**, not a trained action network.  
For each frame we resize it, find the person with **YOLO**, get joints with **MediaPipe**, compute angles and speeds, then apply **if/else rules** to label standing, sitting, walking, running, or lying.  
The Gradio app and the evaluation script use the **same** `MonitorPipeline` class.”

```
frame
  → preprocess.py
  → detector.py (YOLO) + pose.py (MediaPipe) + motion.py (MOG2) + camera_motion.py
  → features.py
  → classifier.py (rules + temporal filter)
  → visualize.py (boxes, skeleton, label)
```

Glue file: **`src/pipeline.py`**

---

# Root entry points

## `app.py` — browser app backend

**What it does:** Starts MotionSense. Loads YOLO **once**, then handles Video Analyze, Photo Analyze, and Live Camera.

**How it works (student words):**
- Imports Gradio, OpenCV, our pipeline.
- `PARAMS = Params.load()` reads `configs/tuned_params.json`.
- `DETECTOR = PersonDetector()` loads YOLO once (warm-up on a blank image).
- `new_pipeline(live=...)` makes a fresh tracker (pose history, MOG2) but **reuses** the shared YOLO.
- `analyse_video` / `analyse_image` / `webcam_step` call `pipe.process(frame, time)`.
- Live webcam uses `live=True` so we skip heavy MOG2/camera-flow for better FPS; Video Analyze stays full pipeline.
- `LOCK` so YOLO isn’t used by two requests at once.

**What I’d say:**  
“`app.py` is the server side of the UI. Models load once at startup. Each click or webcam frame runs the same `MonitorPipeline.process` as evaluation.”

---

## `ui.py` — Gradio layout and CSS

**What it does:** Builds the dark MotionSense interface: sidebar pages, Overview, Video, Photo, Live, About.

**How it works:**
- Big `APP_CSS` string styles the dark theme.
- `build_ui(...)` creates Gradio components and wires buttons to `analyse_video`, etc.
- Live: Camera Preview streams into `webcam_step`, which fills Live Annotated View.
- Sample clips come from `tests/cases/` and `tests/demo_tomorrow/`.

**What I’d say:**  
“`ui.py` is only presentation. All intelligence is still in `src/`. Theme and CSS are passed to `demo.launch()` because Gradio 6 moved them off `Blocks()`.”

---

# Core library — `src/`

## `src/pipeline.py` — the whole system in one class

**What it does:** `MonitorPipeline` runs four stages on every frame.

| Stage | Method | Calls |
|---|---|---|
| 1 | `preprocess()` | `preprocess.py` |
| 2 | `infer()` | YOLO, MediaPipe, MOG2, camera motion |
| 3 | `postprocess()` | features + `classify_pose` + temporal filter |
| 4 | `render()` | `visualize.draw_result` |

**`process(frame, t)`** returns a dict: activity, confidence, people boxes, landmarks, annotated image, timings.

**Live flag:** if `live=True`, skip MOG2 and optical-flow camera compensation (webcam is fixed). Eval/Video keep `live=False`.

**What I’d say:**  
“If you open one file to understand the project, open `pipeline.py`. It’s the story of one frame from pixels to label.”

---

## `src/config.py` — fixed settings

**What it does:** Constants that are **not** tuned as classifier parameters: resize 640, CLAHE, MOG2 history, YOLO conf **0.4**, YOLO every 3 frames, landmark visibility 0.5, paths to models.

**What I’d say:**  
“Hard engineering choices live in `config.py`. The *tunable* rule thresholds live in `params.py` / `tuned_params.json`.”

---

## `src/params.py` — tunable thresholds

**What it does:** Dataclass of sit/walk/run angles and speeds, smoothing window, `require_person_box`, etc.

- `Params.load()` → `configs/tuned_params.json` (chosen on **val** only).
- Defaults = Milestone 2 hand values (for comparison).

**What I’d say:**  
“We didn’t train weights. We tuned physical thresholds on validation with a random search, seed 106.”

---

## `src/preprocess.py` — clean the frame

**What it does:**
1. Resize long side ≤ 640 (live may use 480).
2. Optional CLAHE on Lab L (helps dark rooms).
3. Blurred gray for MOG2 only.

**What I’d say:**  
“YOLO and pose get the sharp color image. Blur is only for background subtraction so noise doesn’t look like motion.”

---

## `src/detector.py` — YOLO person boxes

**What it does:** Loads YOLO11n, predicts COCO class `person`, confidence 0.4, sorts by box area (**largest = primary subject**).

**What I’d say:**  
“YOLO answers *where is the person?* We only classify the largest box if several people appear. We didn’t train YOLO — pretrained COCO weights.”

---

## `src/pose.py` — MediaPipe landmarks

**What it does:**
- MediaPipe Pose Landmarker lite → 33 landmarks with visibility.
- VIDEO mode tracks across frames.
- `pose_on_crop(...)`: if the person is small/missed, run pose on an upscaled YOLO crop.

**What I’d say:**  
“Pose gives joints. Low-visibility points are ignored so we don’t trust guessed off-screen legs.”

---

## `src/features.py` — geometry from landmarks

**What it does:**
- `pose_plausible(...)`: reject hand/face fake skeletons.
- `posture_features(...)`: knee/hip angles, thigh ratio, torso tilt, body height.
- `MotionFeatures`: hip/leg speed in **body-heights per second**, plus stride.

**What I’d say:**  
“Features are scale-free — same rules near or far. Thigh ratio helps front-view sitting when 2D angles look straight.”

---

## `src/classifier.py` — rules + smoothing

**What it does:**
- `classify_pose`: ordered rules — unknown if no hips/legs → lying → sitting → running/walking/standing by speed.
- `classify_blob`: MOG2 baseline (shape + speed only).
- `TemporalFilter`: majority vote + persistence so labels don’t flicker.
- `replay`: re-score cached traces when tuning/evaluating without re-running neural nets.

**What I’d say:**  
“This is our ‘model’ — readable if/else rules, not a neural classifier. That’s why we can explain every decision.”

---

## `src/motion.py` — MOG2 baseline

**What it does:** Background subtraction, blobs, motion events. Used as **baseline** to prove pose+rules are better (macro-F1 ~0.08 vs ~0.77).

**What I’d say:**  
“MOG2 only sees moving pixels. A still sitter fades into the background, so baseline can’t beat pose.”

---

## `src/camera_motion.py` — “is the camera moving?”

**What it does:** Optical flow on background corners + RANSAC → subtract from hip speed so a pan isn’t read as walking/running.

**What I’d say:**  
“Important for HMDB movie clips. Our room demos use a fixed camera, so this matters less live.”

---

## `src/visualize.py` — draw the overlay

**What it does:** Boxes, skeleton, activity text, FPS on the frame for the UI and saved videos.

**What I’d say:**  
“Pure drawing. No classification here.”

---

## `src/evaluate.py` — score val/test

**What it does:**
- Loads segments + cached traces (or runs the pipeline).
- Majority label per segment (ignore no-person/unknown votes).
- Prints macro-F1, accuracy, confusion matrix, writes JSON under `outputs/metrics/`.

**What I’d say:**  
“`python src/evaluate.py --split test` reproduces our report numbers from tag v1.0 using `outputs/cache/`.”

---

## `src/run.py` — CLI for one video/webcam

**What it does:** Command-line runner (annotated mp4 + CSV). Same pipeline as the app.

**What I’d say:**  
“Before Gradio, this was our main demo entry. `predict.py` just forwards here for old commands.”

---

## `src/predict.py` — thin wrapper

**What it does:** Compatibility shim → `src/run.py`.

---

# Configs

## `configs/tuned_params.json`

**What it does:** Saved best thresholds from val tuning (seed 106), including `require_person_box: true`.

**What I’d say:**  
“This file is the frozen rule set for the final numbers. We don’t edit it during the demo.”

Other files: `tuned_params_m3.json` (older), `tuned_params_retune_rejected.json` (looked good on val, worse on test — we rejected it).

---

# Scripts (supporting tools)

| Script | Student explanation |
|---|---|
| `scripts/get_data.py` | Download / prepare public datasets |
| `scripts/import_public.py` | Parse GMDCSA24/HMDB labels into our CSVs |
| `scripts/make_splits.py` | Split by subject/movie (no leakage) |
| `scripts/tune.py` | Random-search thresholds on **val only** |
| `scripts/make_test_cases.py` | Build short `tests/cases` clips |
| `scripts/run_test_cases.py` | Run the TC01–TC13 checklist |
| `scripts/make_slides_m5.js` | Generate `docs/slides_final.pptx` |
| `scripts/make_report_pdf.py` | Markdown → PDF for hand-in |
| `scripts/draw_pipeline.py` | Draw pipeline figure |
| `scripts/failure_gallery.py` | Failure thumbnails for the report |
| `scripts/make_backup_demo.py` | Backup demo reel if live fails |
| `scripts/record_clips.py` | Optional own recordings |

**What I’d say:**  
“Scripts are our factory tools. The live product path is `app.py` → `src/pipeline.py`.”

---

# Data & tests (know the folders)

| Path | What to say |
|---|---|
| `data/` | Splits + annotations; big videos usually not in git |
| `data/README.md` | Data card (GMDCSA24 + HMDB51) |
| `tests/cases/` | Built-in UI samples (TC01–TC13) |
| `tests/demo_tomorrow/` | Extra Analyze pack for presentation |
| `tests/test_cases.csv` | Expected labels for those clips |
| `models/` | YOLO `.pt` + MediaPipe `.task` (downloaded locally) |
| `outputs/cache/` | Cached features so evaluate doesn’t need 3 GB video |
| `outputs/metrics/` | Saved F1 / accuracy JSON |

---

# Docs worth naming in Q&A

| Doc | Use when… |
|---|---|
| `docs/final_report.md` | Formal write-up |
| `docs/design_choices.md` | “Why YOLO / CLAHE / rules not SVM?” |
| `docs/error_analysis.md` | Failure cases |
| `docs/m6/QA_Reviewer_Pack.md` | Short numeric Q&A |
| `docs/m6/Student_Reviewer_Guide.md` | Spoken pitch + Q&A |
| `docs/m6/Student_Codebase_Guide.md` | **This file** — per-file code talk |

---

# If they open a random file — 10-second replies

| They open… | You say… |
|---|---|
| `pipeline.py` | “Four stages: preprocess, infer, postprocess, render.” |
| `classifier.py` | “Rule order: visibility → lying → sitting → run/walk/stand.” |
| `detector.py` | “YOLO person boxes; largest is primary.” |
| `features.py` | “Angles, thigh ratio, body-heights/s; pose gate.” |
| `evaluate.py` | “Segment majority vote; macro-F1; uses cache.” |
| `app.py` | “Gradio handlers; models loaded once.” |
| `ui.py` | “Layout only.” |
| `tune.py` | “Val-only random search for thresholds.” |
| `tuned_params.json` | “Frozen thresholds from val.” |

---

# Practice walk-through (2 minutes)

1. “Start at `app.py` — UI calls `MonitorPipeline`.”  
2. “Open `pipeline.py` — one frame, four stages.”  
3. “`detector.py` + `pose.py` = perception.”  
4. “`features.py` + `classifier.py` = decision.”  
5. “`evaluate.py` + `tuned_params.json` = how we got 0.772 macro-F1.”  
6. “We did not train a deep action model; we tuned explainable rules.”

---

*Tip: keep this PDF beside the laptop during defense. Point at files in the IDE instead of memorizing line numbers.*
