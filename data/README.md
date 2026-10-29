# Data card

The video files are not in git. Rebuild them with the commands in the main README (HMDB51 import), or download
our recordings from the Google Drive link given there.

## Sources and licences
- **HMDB51** (Kuehne et al., "HMDB: A Large Video Database for Human Motion Recognition", ICCV 2011, Brown
  University), classes `walk` and `run`. Free for non-commercial research use. We downloaded the copy at
  huggingface.co/datasets/jili5044/hmdb51 and imported it with
  `python scripts/import_public.py hmdb51 <folder> --limit-per-source 3`.
- **Own recordings** (in progress) by the three team members, made with `scripts/record_clips.py` on laptop
  webcams and phone cameras. Everyone filmed agreed to appear. For course use only.

## Naming and splitting
- `data/raw/<activity>/<source>__<NN>.<ext>`. A **source** is the original video: the movie for HMDB51
  (`hmdb-<movie>`), or one recording session for our own clips (one person, place and camera, e.g.
  `s01-wilmark-room-webcam`).
- The data is **split by source, not by file**. All clips of a source go to the same split, so test never holds
  near-copies of training footage. A movie with both walking and running clips is still one source. The target
  is 70 / 15 / 15 by clip count, with a fixed seed of 106. The assignment is in `data/splits.csv`.
- **The test split is locked.** Re-running `make_splits.py` only places new sources. We do not look at test
  results while tuning thresholds.

## Labels
- One activity per clip, taken from the folder name: `standing`, `sitting`, `walking`, `running`, `lying`.
- **HMDB51:** the original HMDB labels (manually annotated by the dataset authors). We mapped `walk → walking` and
  `run → running`. Using HMDB's own metadata in the file names, we kept only clips with the **full body visible**
  (`f`), **one person** (`np1`) and quality not `bad`, and at most 3 clips per movie so no film dominates.
  Of 780 walk/run clips, 294 were kept.
- **Own recordings:** the activity is chosen before recording and done for the whole clip, after a 3-second
  countdown. A second member watches each clip and deletes any that mix activities.
- HMDB51 `sit` and `stand` were **not used**: they show the *motion* of sitting down or standing up, not the
  posture.
- `data/annotations/motion_counts.csv` will hold hand-counted motion events for 3 static-camera clips, for the
  MOG2 baseline. An event is the person starting to move after being still.

## Counts

| split | standing | sitting | walking | running | lying | total | sources |
|-------|---------:|--------:|--------:|--------:|------:|------:|--------:|
| train | 0 | 0 | 127 | 78 | 0 | 205 | 79 |
| val   | 0 | 0 | 32 | 12 | 0 | 44 | 20 |
| test  | 0 | 0 | 27 | 18 | 0 | 45 | 20 |
| **total** | 0 | 0 | 186 | 108 | 0 | **294** | **119** |

Standing, sitting and lying will be filled with our own recordings, which are being added as new sources.

## Image properties
- HMDB51 clips are **240 px tall**, 320–560 px wide (320×240 is the most common), at **30 fps**. They last
  0.8–17.8 s, with a median of 2.7 s, for 15 minutes in total.
- Webcams record 640×480 or 1280×720, and phones 1920×1080.
- Every frame is resized so the long side is at most 640 before processing (`src/preprocess.py`). HMDB51 clips are
  already smaller, so they are not resized.

## Preprocessing applied (in `src/preprocess.py`, shared by prediction and evaluation)
1. Resize so the long side is at most 640 (INTER_AREA).
2. CLAHE on the L channel of Lab, with clip 2.0 and an 8×8 grid, for low light.
3. Grayscale and a 5×5 Gaussian blur, applied only to the input of MOG2.
No augmentation is applied because nothing is trained. The rule thresholds are tuned on `val` only.
See `docs/figures/preprocessing.jpg`.

## Known gaps
- **No standing, sitting or lying clips yet.** This is the main gap until our own recordings are in.
- **Moving cameras:** most HMDB51 clips are movie shots where the camera follows the actor. That breaks
  frame-relative speed and background subtraction.
- **Low resolution and short clips:** 240 px tall and often under 3 s, while the pose model and the 0.5 s speed
  window need a few seconds.
- Mostly daylight scenes: there is little night or backlit footage. Mostly side and back views.
- One person per clip. With several people in frame, we track the largest (closest) one.
- `lying` is only a posture. Falls are not detected yet.
