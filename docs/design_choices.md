# Design choices and why

Fixed settings live in [src/config.py](../src/config.py). The rule thresholds are **parameters**
([src/params.py](../src/params.py)): they are tuned on the **val** split only by `scripts/tune.py`, and the result is
saved to [configs/tuned_params.json](../configs/tuned_params.json).

## 1. Preprocessing ([src/preprocess.py](../src/preprocess.py))

| Step | Choice | Why |
|---|---|---|
| Resize | Long side 640 px, `INTER_AREA` | YOLO11n was trained at 640, so a larger input only costs time. 640 keeps a full-body person 150–400 px tall, enough for MediaPipe. A 1080p phone frame has about 5× fewer pixels after resizing. At 640 we measured 7–10 FPS on an i7-6700HQ laptop CPU. `INTER_AREA` averages pixels when shrinking, so it does not alias. |
| Contrast | **CLAHE** on the L channel of Lab, clip 2.0, 8×8 tiles | Webcams in dim rooms give flat, dark frames where pose detection fails. **Global** histogram equalisation stretches the whole image at once, so a bright window makes the person darker. CLAHE equalises each tile locally. The clip limit stops it from amplifying sensor noise. We use Lab so only lightness changes: running it on BGR channels would shift skin and clothing colours. |
| Grayscale + Gaussian blur 5×5 | Input to MOG2 only | Background subtraction compares each pixel with its history. Colour adds little but triples the work. The blur removes per-pixel sensor noise that MOG2 would otherwise mark as motion. Gaussian is cheap, and 5×5 is the smallest kernel that cleaned the noise in our tests. Pose and YOLO receive the **unblurred** colour image, because blur removes the edges they need. |
| No augmentation | – | Nothing is trained. We use pretrained YOLO and MediaPipe plus hand-written rules, so there is nothing to augment for. |

**"What if the video is darker?"** CLAHE raises local contrast, which helps pose find the joints in dim light. Check this with the dim-light clips in val and compare against `--no-enhance`. If MediaPipe's landmark visibility drops below 0.5, the system says `unknown` or `no person` instead of guessing. MOG2 becomes noisier in the dark because sensor noise grows. The 5×5 blur, the morphological opening and the 0.4% minimum blob area filter most of it out.

## 2. Motion: MOG2 background subtraction ([src/motion.py](../src/motion.py))

- **MOG2** (Gaussian mixture per pixel) instead of plain frame differencing. Frame differencing only sees the *edges* of moving objects and misses a person who stops. MOG2 learns a model of the background, so the whole person is foreground. It also adapts to slow lighting changes.
- `history=300` (≈10 s at 30 fps): long enough that someone sitting still is not instantly absorbed into the background.
- `varThreshold=25`: how many squared standard deviations a pixel must move to count as foreground. Lower values are more sensitive and catch shadows and noise.
- `detectShadows=True` + threshold at 200: shadows are labelled 127 and dropped, so a passing shadow does not trigger motion. This was one of the problems in our proposal's problem statement.
- Opening then closing (5×5 ellipse): opening deletes speckles, closing fills holes in the body silhouette.
- Minimum blob = 0.4% of the frame: ignores small things such as a moving curtain or a pet far away.
- **Motion event** = at least 5 moving frames after at least 15 still frames. This turns a flickering per-frame flag into countable events that we compare with a hand count.

## 3. Person detection: YOLO11n ([src/detector.py](../src/detector.py))

- **Pretrained, not trained by us.** COCO already has a `person` class trained on more than 60k images, and our dataset is far too small to beat that.
- **Nano** model: the smallest and fastest, and accurate enough for people who fill a good part of the frame.
- `conf=0.4`: lower values gave false boxes on chairs and coats, higher values missed people who were partly hidden.
- Runs every 3rd frame and reuses the boxes in between. People move little in 0.1 s, and this roughly doubles the FPS.
- The **largest** box is the primary subject, because the person closest to the camera is the one being monitored. This is the fallback from risk 1 in the proposal.

## 4. Pose: MediaPipe Pose Landmarker (lite) ([src/pose.py](../src/pose.py))

- Gives 33 body landmarks with a visibility score. The alternative is classifying raw pixels, which would need thousands of labelled frames and would break with new clothes, rooms or lighting. Joint positions are the same whatever you wear.
- **VIDEO** running mode: it tracks the person between frames instead of detecting from scratch, which is faster and steadier.
- Landmarks with visibility below 0.5 are ignored. MediaPipe still *guesses* coordinates for joints outside the frame, and trusting those labelled a person seated at a desk as "standing".

## 5. Features ([src/features.py](../src/features.py))

All features are **angles or ratios**, so they do not change with distance from the camera or image size.

| Feature | Meaning | Used for |
|---|---|---|
| Knee angle (hip–knee–ankle) | ≈180° straight leg, ≈90° bent | sitting (side view) |
| Hip angle (shoulder–hip–knee) | ≈180° upright, ≈90° folded | sitting (side view) |
| Thigh ratio = vertical hip→knee drop ÷ torso length | ≈1 standing, near 0 when the thigh points at the camera | sitting (**front** view, where the angles look straight in 2D) |
| Torso angle from vertical | 0° upright, 90° horizontal | lying |
| Hip speed, in body-heights per second | how fast the body travels across the frame | walking / running sideways |
| Leg swing, in body-heights per second | range of ankle positions relative to the hips over 0.5 s | walking / running **toward** the camera, where the hips barely move sideways |

Speeds are divided by body height so a person far away (small in pixels) and a person close up give the same number. Leg swing uses the *range* over the window, not the summed path, because summing frame-to-frame jitter labelled walkers as runners.

### Camera-motion compensation ([src/camera_motion.py](../src/camera_motion.py)), added in Milestone 3

In Milestone 2, 11 of 32 walking clips were called "standing". These were movie shots where **the camera follows
the actor**, so the hips stay in the same place *in the frame*. The fix:
1. Find up to 150 corner points (`goodFeaturesToTrack`) on the background, with the person's box (+15%) masked out.
2. Track them into the next frame with **Lucas–Kanade optical flow** and take the **median** shift. The median
   ignores the few points that land on other moving things.
3. Add up those shifts into a camera offset, and compute `hip_speed_comp` = hip travel **minus** camera travel.
   When the camera pans right to follow a walker, the background moves left, and subtracting it gives back the
   walker's real speed.
4. If fewer than 12 points can be tracked (a plain wall, darkness), the shift is taken as 0.
Whether to use `hip_speed_comp` instead of `hip_speed` is itself a parameter (`camera_compensation`), so tuning
decides whether it helps. Tuning turned it on, but its effect is small: on val, switching it off changes 1 of 108
segments. See [error_analysis.md](error_analysis.md).

## 6. Classifier ([src/classifier.py](../src/classifier.py))

Rules are checked in this order. The first match wins (the values are in `configs/tuned_params.json`; the names
are the fields in `src/params.py`):
1. Hips not visible → `unknown`
2. Torso tilted ≥ `lying_torso_angle_min` → `lying`
3. No thigh visible → `unknown`
4. Leg folded (knee < `sit_knee_angle_max` and hip < `sit_hip_angle_max`, or thigh ratio < `sit_thigh_ratio_max`)
   **and** slow → `sitting`
5. Hip speed ≥ `run_hip_speed_min` or leg swing ≥ `run_leg_speed_min` → `running`
6. Hip speed ≥ `walk_hip_speed_min` or leg swing ≥ `walk_leg_speed_min` → `walking`
7. Otherwise → `standing`

**Post-processing** ([`TemporalFilter`](../src/classifier.py)):
- **Majority vote** over the last `smoothing_window` frames removes one-frame flicker. This is the "temporal
  smoothing" from risk 1 in the proposal.
- **Persistence:** the shown label only changes once a new label has won `persist_frames` votes in a row, and
  `sit_persist_frames` for sitting. A running stride bends the knee below the sitting angle for a few frames, so
  sitting has to last longer before it is shown. This is the guide's "a detection must persist for N frames".
- `no person` / `unknown` frames keep the last label instead of resetting it.

**Why rules and not an SVM?** We have no labelled frame-level training data of our own, only clip and segment
labels. Every rule can be explained and fixed separately, and the thresholds are physical quantities (degrees,
body-heights per second) that we can check on a single frame. With 13 parameters and about 200 val segments,
tuning is much less likely to overfit than training a classifier.

### Baseline: MOG2 blob rules

To show what pose adds, the baseline sees **only** the MOG2 foreground. It takes the largest moving blob's shape
(height/width) and the speed of its centre (blob-heights per second): wide → lying, squarish → sitting, tall →
standing, walking or running by speed. If there is no blob, the person is taken to be standing still (MOG2 absorbs
a still person into the background). It is tuned on val with the same number of trials as the main method, so the
comparison is fair.

### Tuning ([scripts/tune.py](../scripts/tune.py))

Random search with 1500 trials per method, seed 106. Each trial draws a full parameter set within physically
sensible ranges (e.g. sitting knee angle between 100° and 160°), replays the **cached val features** through the
rules, and keeps the set with the best **macro-F1** (accuracy breaks ties). The neural networks are not re-run, so a
trial takes milliseconds. The **test split is never read**: `tune.py` only calls `evaluate("val")`.

## 7. Evaluation choices

- **Split by source:** the movie for HMDB51, the subject for GMDCSA24. Frames from one recording are
  near-duplicates, so if they were in both train and test, every number would be too high. Each dataset is split
  on its own so every split gets both kinds of footage. For GMDCSA24 (4 subjects), we chose **before computing any
  result on them** that test gets a subject filmed in both day and night (subject 3). Subject 1 was only filmed in
  daylight, so putting it in test would have left out the hardest condition on purpose.
- **Segments, not clips:** a GMDCSA24 video contains several activities with time ranges ("Sitting[0 to 3];
  Walking[3 to 6]"). Each range is scored separately, as the majority of its frame labels. Frames that also fall
  inside a range with a *different* label (transitions) are left out. Ranges shorter than 1 s are skipped.
- **Macro-F1:** the average F1 across classes, so a rare class such as `lying` (few segments) counts as much as a
  common one. Accuracy alone would hide a class that always fails.
- **Same frames for every method:** each video goes through the full pipeline once, and the baseline, the M2 rules
  and the tuned main method are all scored from the same cached features.
- **Speed** is the median time per frame from preprocessing to overlay, excluding video decoding. It is measured
  after a 30-frame warm-up per clip, with YOLO on, on the named laptop CPU.

## Likely questions, by owner

- *Why 640 and not the full resolution?* See section 1: it is YOLO's training size, and it means about 5× fewer pixels than 1080p.
- *Why CLAHE and not plain equalizeHist?* Local vs global: a bright window would darken the person.
- *Why MOG2 and not frame differencing?* A person who stops would vanish with differencing.
- *What does varThreshold do?* It sets how far a pixel must move from its background model to count as foreground.
- *Why pose angles and not raw pixels?* They don't depend on clothing, lighting or distance, and they need no training data.
- *Why is the thigh ratio needed?* From the front, a seated person's knee angle still looks about 180° in 2D.
- *Which split did this number come from?* `val`. Test is locked in `data/splits.csv` and only used for the final evaluation.
- *What happens with two people?* We track the largest one. Others get grey boxes and no label.
