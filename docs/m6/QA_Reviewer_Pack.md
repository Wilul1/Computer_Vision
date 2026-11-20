# Milestone 5–6 Q&A — Reviewer Pack

Study this out loud. Numbers are from tag **`v1.0`**: test macro-F1 **0.772**, accuracy **0.756**, **15.1 FPS**, room **93%**, movie **58%**, baseline macro-F1 **0.081**.

**Honest fallback line (use when unsure):**  
“I’m not sure of the exact number, but I would check by changing that setting on val and comparing the annotated output / confusion matrix.”

---

# Part A — Team presentation Q&A (everyone)

### A1. Why is test lower/higher than validation?
**Answer:** Val and test are different sources (different subjects / movies). We tune only on val. Test includes hard HMDB moving-camera clips. Also M4 fixes were informed by earlier test failures, so the big test jump is a bit optimistic; the safer story is the **val** gain (+0.066 balanced F1). We never retune on test.

### A2. Why macro-F1 instead of accuracy?
**Answer:** Classes are imbalanced — lying has **3** test segments, walking **36**. Accuracy can look fine while ignoring rare classes. Macro-F1 averages F1 per class equally, so lying and running still matter.

### A3. What does the confusion matrix tell you?
**Answer:** Sitting is strong (recall 1.00). Running is weakest (recall 0.50) — mostly movie clips. Standing often absorbs slow walkers (precision 0.545). That matches our limitation: slow walk → standing.

### A4. How would it behave at night / twice the distance / different camera?
**Answer:**  
- **Night with lights on:** room night is still **93%** — CLAHE helps.  
- **Near darkness:** often **no person** (YOLO/pose fail) — TC09.  
- **Twice the distance:** person &lt; ~100–150 px → pose/YOLO weak (small-person accuracy ~57%).  
- **Hand-held / panning:** bad — camera motion estimate breaks (58% on movies). Designed for a **fixed** camera, full body, ~2–5 m.

### A5. What is IoU and which threshold did you use?
**Answer:** IoU = intersection-over-union of predicted box vs ground-truth box. We did **not** train a detector or score activity with IoU. YOLO uses a **confidence** threshold **0.4**, not an IoU threshold. For activity we score **segment majority labels**, not box IoU.

### A6. Why this preprocessing (640, CLAHE, blur)?
**Answer:** 640 matches YOLO’s training size and keeps CPU real-time. CLAHE on Lab **L** raises local contrast in dark rooms without shifting colours. Gaussian blur is **only** for MOG2 (noise); pose/YOLO get the unblurred colour frame.

### A7. Did you train a model?
**Answer:** No. YOLO11n and MediaPipe are pretrained. We only **tune rule thresholds** on val (random search, seed 106).

### A8. Why not a neural action classifier?
**Answer:** Few hundred segments, not frame labels; we need CPU real-time and explainable rules (angles, body-heights/s). Rules overfit less here than training a classifier on this size.

### A9. What about falls / sudden abnormal motion from the proposal?
**Answer:** Not implemented. We classify lying, but not “fast drop → fall.” Documented as future work. Scope focus: everyday activities with measurable accuracy.

### A10. How do you reproduce the numbers?
**Answer:** `python src/evaluate.py --split test` from `v1.0`. Uses committed `outputs/cache/` — no need for the 3 GB videos.

---

# Part B — Wilmark defense (pipeline, rules, eval, app)

**Files they may open:** `src/classifier.py`, `src/features.py`, `src/evaluate.py`, `src/pipeline.py`, `app.py`, `scripts/tune.py`

## Walk-through (practice saying this)

**`src/pipeline.py`:** one frame → preprocess → YOLO + pose + camera motion + MOG2 → features → rules → draw overlay. Same path for app and evaluate.

**`src/classifier.py`:** ordered rules — no hips → unknown; torso tilt → lying; folded leg / thigh ratio while still → sitting; then run/walk/stand by speed + stride; majority vote + persistence.

**`src/evaluate.py`:** loads segments, majority vote over frames (ignore no-person), prints baseline vs main, writes metrics JSON + confusion matrices.

## “What if” (Wilmark)

| Question | Predict | Why |
|---|---|---|
| Double Gaussian blur kernel (5→11)? | MOG2 baseline worse; main almost unchanged | Blur only feeds MOG2; pose/YOLO use sharp colour |
| Drop CLAHE? | More misses in dark rooms; night accuracy down | Local contrast helps landmarks/boxes |
| YOLO conf 0.4 → 0.2? | More false boxes (chairs/coats); wrong “largest person” | Lower conf = more false positives |
| YOLO conf → 0.7? | More “no person”; lying/dark worse | Misses true people |
| Turn off `require_person_box`? | TC13 may fail again (fake skeleton); maybe recover TC04/09 | Gate removes hallucinations and true hard poses |
| Speed window 0.5 s → 2 s? | Labels lag more on sit↔stand; short segments wrong | Longer memory of old motion |
| Tune on **test**? | Inflated dishonest score | Test must stay frozen |
| 60 FPS video instead of 30? | Same rules if speed is in body-heights/s; more CPU load | Time is in seconds, not frames; YOLO every-3-frames denser in time |
| App reload model every frame? | ~1 FPS, unusable | We load YOLO once at startup |

## Concept questions (Wilmark)

**Q: What is overfitting here?**  
A: Thresholds that fit val quirks (or one dataset) and fail on test. Example: pooled macro-F1 overfit to movie walkers; sitting broke. Rejected retune file scored high on val, worse on test.

**Q: Why a validation set?**  
A: Choose thresholds / design changes without touching test. Test is the final report number once.

**Q: What does a convolution layer learn? (YOLO uses CNNs)**  
A: Early layers: edges/textures; deeper: parts/objects. YOLO’s backbone is CNN features → boxes. We didn’t train it; we use its person boxes.

**Q: Background subtraction vs pose?**  
A: MOG2 = moving pixels (baseline). Pose = joint geometry even when sitting still. That’s why baseline F1 is 0.081 and main is 0.772.

---

# Part C — Porcopio defense (detect, pose, data, tests, docs)

**Files they may open:** `src/detector.py`, `src/pose.py`, `src/camera_motion.py`, `scripts/make_splits.py`, `scripts/import_public.py`, `tests/test_cases.csv`

## Walk-through

**`src/detector.py`:** load YOLO11n; predict class person; conf 0.4; sort by box area (largest = primary subject).

**`src/pose.py`:** MediaPipe 33 landmarks; ignore low visibility; if person short/missed, pose on YOLO crop upscaled to 256 px.

**`src/camera_motion.py`:** LK optical flow on background corners (person masked); fit shift/rotation/zoom with RANSAC; subtract from hip speed.

**Splits:** by subject/movie in `data/splits.csv` — no leakage.

## “What if” (Porcopio)

| Question | Predict | Why |
|---|---|---|
| Pose crop threshold 120 → 60 px? | Fewer crop runs; more lost landmarks on medium people | Crop helps when full-frame pose fails |
| Crop target 256 → 128? | Worse landmarks on small people | Less detail after upscale |
| Landmark visibility 0.5 → 0.2? | Trust guessed joints off-screen → desk sit looks “standing” | MediaPipe invents low-visibility points |
| No RANSAC (median shift only)? | Movie/pan clips worse | Median can’t model zoom/rotation well |
| Random split instead of by source? | Test score looks higher, dishonest | Same person/movie leaks into test |
| YOLO every frame instead of every 3? | Slightly stabler boxes, ~slower FPS | People move little in 0.1 s |
| Phone portrait video? | Works if full body visible (TC11 pass) | Pipeline resizes long side ≤640 |

## Concept questions (Porcopio)

**Q: What is non-maximum suppression (NMS)?**  
A: After a detector proposes many overlapping boxes for the same object, NMS keeps the highest-confidence box and suppresses overlaps with high IoU. YOLO does this internally; we then take the largest remaining person box.

**Q: What is optical flow?**  
A: Estimate where pixels moved between frames. We track background corners to estimate **camera** motion so person hip speed isn’t polluted by pans.

**Q: What does RANSAC do?**  
A: Fit a model (here camera motion) while ignoring outlier matches (e.g. moving leaves). Better than a plain average when many points are wrong.

**Q: HOG / Otsu — do you use them?**  
A: Not in our pipeline. HOG = gradient histograms for detection/recognition; Otsu = automatic threshold on a grey histogram. We use YOLO + CLAHE + rules instead. (Know the definition if asked comparatively.)

---

# Part D — Mini mock review (take turns)

**Reviewer asks → student answers in ≤30 seconds.**

1. Show me where sitting is decided. → `classifier.py` / thigh ratio + knee-hip angles before speed.  
2. Why 15 FPS not 30? → Infer (YOLO+pose) ~54 ms of 66 ms/frame on CPU.  
3. One failure and its cause. → TC08 arm exercise → walking; or T048 moving camera.  
4. What would you deploy for real? → Fixed camera, full body, room light, alert on lying/fall (fall not built), edge box, not hand-held.  
5. Who owns detector.py? → Porcopio. Who owns evaluate.py? → Wilmark.

---

# Cheat sheet (memorize)

| Fact | Value |
|---|---|
| Main metric | macro-F1 **0.772** |
| Accuracy | **0.756** |
| Baseline F1 | **0.081** |
| Speed | **66 ms · 15.1 FPS** |
| Room / movie | **93% / 58%** |
| YOLO conf | **0.4** |
| Resize | long side **640** |
| Tune on | **val only**, seed **106** |
| Not built | **falls** |
| Reproduce | `python src/evaluate.py --split test` |
