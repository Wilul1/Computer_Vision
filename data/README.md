# Data card

> Fill in the `TODO`s once the clips are recorded. `scripts/make_splits.py` prints the counts.

## Sources and licences
- **Own recordings** by the three team members, made with `scripts/record_clips.py` on laptop webcams and phone
  cameras. Everyone filmed agreed to appear. For course use only.
- TODO: list any public dataset clips here with their licence, or write "none".

## Naming and splitting
- `data/raw/<activity>/<source>__<NN>.mp4`. A **source** is one recording session: one person, one place, one
  camera, e.g. `s01-wilmark-room-webcam`.
- The data is **split by source, not by file**. All clips of a session go to the same split, so test never holds
  near-copies of training footage. The target is 70 / 15 / 15 by clip count, with a fixed seed of 106. The
  assignment is recorded in `data/splits.csv`.
- **The test split is locked.** Re-running `make_splits.py` only places new sources. We do not look at test
  results while tuning thresholds.

## Labels
- One activity per clip, taken from the folder name: `standing`, `sitting`, `walking`, `running`, `lying`.
- Each clip is recorded while doing only that activity, starting after the 3-second countdown, so the label is
  known when filming. A second member watches each clip and deletes any that mix activities.
- `data/annotations/motion_counts.csv` holds hand-counted motion events for 2–3 clips, used by the MOG2 baseline.
  An event is the person starting to move after standing still.

## Counts (TODO: paste the output of make_splits.py)

| split | standing | sitting | walking | running | lying | total | sources |
|-------|---------:|--------:|--------:|--------:|------:|------:|--------:|
| train |          |         |         |         |       |       |         |
| val   |          |         |         |         |       |       |         |
| test  |          |         |         |         |       |       |         |

## Image properties
- Webcams usually record 640×480 or 1280×720 at 30 fps. Phones record 1920×1080 at 30 fps.
- Everything is resized to a long side of 640 before processing (`src/preprocess.py`).

## Preprocessing applied (in `src/preprocess.py`, shared by prediction and evaluation)
1. Resize so the long side is 640 (INTER_AREA).
2. CLAHE on the L channel of Lab, with clip 2.0 and an 8×8 grid, for low light.
3. Grayscale and a 5×5 Gaussian blur, applied only to the input of MOG2.
No augmentation is applied because nothing is trained. The rule thresholds are tuned on `val` only.

## Known gaps
- TODO after recording: few night or backlit scenes? Only indoor scenes? Mostly side views?
- One person per clip. With several people in frame, we track the largest (closest) one.
- `lying` is only a posture. Falls are not detected yet.
