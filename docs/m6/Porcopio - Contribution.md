# Individual Contribution Statement — Milestone 6

**Porcopio Zheniel Ivan** · CSST 106 Computer Vision · Laguna State Polytechnic University  
Team: Porcopio Zheniel Ivan · Veridiano Wilmark  
Repository: https://github.com/Wilul1/Computer_Vision · tag `v1.0`

**Note on git history.** Commits were pushed from the shared team account. My modules and milestones are listed below with paths and the commits where those parts entered the tree; pair sessions are noted in both statements (Appendix C of the final report).

## 1. What I built

| Area | Paths |
|---|---|
| Person detection | `src/detector.py` (YOLO11n) |
| Pose estimation | `src/pose.py` (MediaPipe Pose, 33 landmarks; crop fallback) |
| Camera-motion compensation | `src/camera_motion.py` (LK flow + RANSAC) |
| Dataset import & splits | `scripts/import_public.py`, `scripts/make_splits.py`, `data/splits.csv`, `data/README.md` |
| Test cases & backup demo | `tests/test_cases.csv`, `tests/cases/`, `scripts/make_test_cases.py`, `scripts/run_test_cases.py`, `docs/backup_demo_m4.mp4` |
| Report / limitations / slides | `docs/final_report.md`, `docs/limitations.md`, `docs/slides_final.pptx` / `.pdf` |

**Key commits (3–5):**

1. [725543a](https://github.com/Wilul1/Computer_Vision/commit/725543a) — Milestone 2: detector + pose integrated in the first prototype  
2. [7fc9eb2](https://github.com/Wilul1/Computer_Vision/commit/7fc9eb2) — Public dataset importer and split-by-source design  
3. [4e48885](https://github.com/Wilul1/Computer_Vision/commit/4e48885) — HMDB51 split + val reporting  
4. [cf19105](https://github.com/Wilul1/Computer_Vision/commit/cf19105) — Milestone 4: RANSAC camera motion, crop pose, test-case pack from M3 failures  
5. [6863b05](https://github.com/Wilul1/Computer_Vision/commit/6863b05) / [077adae](https://github.com/Wilul1/Computer_Vision/commit/077adae) — Final report/slides package and backup demo reel  

## 2. A decision I made

**Choice:** split the data **by source** (GMDCSA24 **subject**, HMDB51 **movie**), never by random frames or clips from the same person/film across train/val/test.

**Rejected:** random clip shuffle or frame-level split (easier counts, but leaks appearance and motion style into test).

**Evidence:** the same subject or movie in train and test would inflate accuracy without proving generalization. With a source split, fixed-room test accuracy is **93%** while moving-camera movie clips are only **58%**—an honest gap we report instead of hiding. The frozen lists live in `data/splits.csv`. Test subject (day + night) was chosen **before** computing test metrics.

## 3. A problem I solved

**Problem:** MediaPipe sometimes drew a **confident skeleton on an empty / patterned background** (curtain), so “no person” photos could still get a fake pose; also small people in movie frames lost landmarks.

**Cause:** pose can hallucinate joints when visibility is weak; full-frame pose on a tiny subject fails.

**Fix:** (1) run pose on the **YOLO crop upscaled to 256 px** when the person is short or missed; (2) with the team, gate trusted poses behind a **YOLO person box** so empty-room TC13 passes. Side effect we document: lying on the floor and near-dark clips (TC04, TC09) can become “no person” when YOLO misses—accepted on val for +0.024 balanced F1. Camera pans were a separate failure mode; switching camera compensation from median shift to **RANSAC** helped on val (+0.026) but does not fix fast blurred pans (e.g. T048).

## 4. What I learned

1. **IoU / detection confidence vs tracking** — YOLO’s box confidence is not the same as “correct activity”; we use boxes to find the person and gate pose, then classify with geometry. Lowering conf too far (e.g. 0.2) would add chair/coat boxes and confuse the “largest person” rule.  
2. **Optical flow and camera motion** — hip speed in image coordinates mixes person motion and camera motion; subtracting a RANSAC camera model is why movie clips are usable at all, and why fast pans still break us.  

**One thing I would do differently:** label more **lying** segments early (only 3 val / 3 test) and collect a small fixed-webcam set from our own rooms before the 80% check, so demo conditions match the metrics. Fall detection (fast torso drop → lying) remains future work from the proposal.
