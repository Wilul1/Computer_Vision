# Data card

The video files are not in git. Rebuild them with the commands in the main README; `data/splits.csv` is committed,
so every machine gets the same split.

## Sources and licences
- **GMDCSA24** (Alam et al., "GMDCSA-24: A dataset for human fall detection in videos", *Data in Brief* 2024),
  **MIT licence**, from github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos. One room, a
  **fixed camera**, 4 subjects, **day, night and lights-off** recordings, 1280×720 at about 30 fps. 160 videos
  (79 daily activities, 81 falls). This is the closest public match to our use case: a fixed camera watching a room.
- **HMDB51** (Kuehne et al., ICCV 2011, Brown University), classes `walk` and `run`, for non-commercial research.
  Copy from huggingface.co/datasets/jili5044/hmdb51. Movie and web clips, often with a **moving camera**.
- **Own recordings**: not recorded yet. `scripts/record_clips.py` is ready, and new sources join the split
  without moving existing ones.

## Labels
- **GMDCSA24:** each video comes with the authors' **time-segment labels**, e.g. `Sitting[0 to 3.4]; Walking[3.4 to
  6]`. We map Sitting → sitting, Sleeping → lying, Walking → walking, Standing → standing. Falling, Reading,
  Drinking, Eating, Writing and Exercising are not our classes and are ignored; Reading etc. overlap a posture label
  anyway. `scripts/import_public.py` parses the CSVs. It tolerates rows with several ranges in one bracket and
  missing `]`, both of which occur. One row has an impossible range (`9.7 to 3`, subject 2 ADL 06), and it is
  dropped. Segments go to `data/annotations/segments.csv`.
- **HMDB51:** the dataset authors' clip labels. We keep only full-body (`f`), single-person (`np1`), non-`bad`
  clips, at most 3 per movie (294 of 780). HMDB's `sit` and `stand` are the motions of sitting down and standing
  up, not postures, so they are not used.
- Each clip's recording **condition** (day/night, static or moving camera, view, quality, description) is in
  `data/annotations/clip_info.csv` and appears in the test log.

## Splitting (by source, no leakage)
- Source = **movie** for HMDB51, **subject** for GMDCSA24. All clips of a source are in one split.
- Each dataset is split on its own (`scripts/make_splits.py`), so each split has both kinds of footage.
- GMDCSA24 has only 4 subjects. **Before computing any result on them**, we chose test = subject 3 and
  val = subject 2, because both were filmed in day **and** night; subject 1 is daylight only. Train = subjects 1
  and 4. Putting the daylight-only subject in test would have left the hardest condition out of the test set.
- **The test split is locked** and was used once, for the final Milestone 3 table, after tuning on val.
- The method has no trained model (pretrained YOLO and MediaPipe plus rules). **Train is not used** for fitting. It
  is kept as a reserve for a learned classifier later. Thresholds are tuned on **val**.

## Counts (labelled segments ≥ 1 s that are scored)

| split | standing | sitting | walking | running | lying | segments | GMDCSA videos | HMDB clips | sources |
|-------|---------:|--------:|--------:|--------:|------:|---------:|--------------:|-----------:|--------:|
| train | 7 | 37 | 149 | 78 | 9 | 280 | 60 (2 subjects) | 205 | 81 |
| val   | 23 | 15 | 55 | 12 | 3 | 108 | 44 (subject 2) | 44 | 21 |
| test  | 15 | 18 | 36 | 18 | 3 | 90 | 37 (subject 3) | 45 | 21 |

GMDCSA videos with no posture segment of at least 1 s (e.g. a fall straight from the start) are not counted.

## Image properties
- GMDCSA24: 1280×720, about 30 fps, 3–17 s per video. HMDB51: 240 px tall, 320–560 px wide, 30 fps, median 2.7 s.
- Every frame is resized so the long side is at most 640 before processing (`src/preprocess.py`).

## Preprocessing applied (in `src/preprocess.py`, shared by run.py, evaluate.py and tune.py)
1. Resize so the long side is at most 640 (INTER_AREA).
2. CLAHE on the L channel of Lab, with clip 2.0 and an 8×8 grid, for low light.
3. Grayscale and a 5×5 Gaussian blur, used for MOG2 and the camera-motion optical flow.
No augmentation, because nothing is trained. See `docs/figures/preprocessing.jpg`.

## Known gaps
- **Lying: 3 val and 3 test segments.** Any lying number is unreliable. The lying-on-the-floor part at the end of
  the fall videos is not labelled by the authors.
- **Standing: only 7 train segments**, which doesn't matter now because train is unused.
- Only 4 people in the room footage, one room, and one camera angle per subject.
- HMDB51 clips are low resolution and short, and two-thirds have a moving camera.
- No footage of our own yet (laptop webcam or phone), which is the real use case.
