# Design choices and why

Every number here lives in [src/config.py](../src/config.py). Thresholds are tuned on the **val** split only.

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

## 6. Classifier ([src/classifier.py](../src/classifier.py))

Rules are checked in this order. The first match wins:
1. Hips not visible → `unknown`
2. Torso tilted ≥ 60° → `lying`
3. No thigh visible → `unknown`
4. Leg folded (knee < 130° and hip < 140°, or thigh ratio < 0.45) **and** slow → `sitting`
5. Hip speed ≥ 1.2 or leg swing ≥ 2.2 → `running`
6. Hip speed ≥ 0.25 or leg swing ≥ 0.6 → `walking`
7. Otherwise → `standing`

Then a **majority vote over the last 9 frames** (≈0.3 s) removes one-frame flicker. This is the "temporal smoothing" from risk 1 in the proposal.

**Why rules and not an SVM?** With a few dozen clips, a trained classifier would overfit. Every rule can be explained and fixed separately, and the thresholds are physical quantities we can measure. An SVM on the same features is an option for Phase 3 once we have more data, and we would compare it against these rules on val.

## 7. Evaluation choices

- **Split by source** (person + place + camera), 70/15/15. Frames from one recording are near-duplicates. If they were in both train and test, every number would be too high.
- **Macro-F1**: the average F1 across classes, so a rare class such as `lying` counts as much as a common one. Accuracy alone would hide a class that always fails.
- **Clip-level label** = majority of the per-frame labels in the clip, excluding `no person` and `unknown`. Each clip has one true activity.
- **FPS** is measured inside the pipeline (preprocessing to overlay), excluding video decoding.

## Likely questions, by owner

- *Why 640 and not the full resolution?* See section 1: it is YOLO's training size, and it means about 5× fewer pixels than 1080p.
- *Why CLAHE and not plain equalizeHist?* Local vs global: a bright window would darken the person.
- *Why MOG2 and not frame differencing?* A person who stops would vanish with differencing.
- *What does varThreshold do?* It sets how far a pixel must move from its background model to count as foreground.
- *Why pose angles and not raw pixels?* They don't depend on clothing, lighting or distance, and they need no training data.
- *Why is the thigh ratio needed?* From the front, a seated person's knee angle still looks about 180° in 2D.
- *Which split did this number come from?* `val`. Test is locked in `data/splits.csv` and only used for the final evaluation.
- *What happens with two people?* We track the largest one. Others get grey boxes and no label.
