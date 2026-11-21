# Student Reviewer Guide — What We Built & How to Explain It

**Project:** MotionSense / Activity Monitor (CSST 106)  
**Team:** Porcopio Zheniel Ivan · Veridiano Wilmark  
**Repo:** https://github.com/Wilul1/Computer_Vision · tag **v1.0**  
**App:** `.\.venv\Scripts\python.exe app.py` → http://127.0.0.1:7860  

Use this like a study script. Say it in your own words — don’t memorize every sentence.

**If you freeze:**  
“I’m not sure of the exact number off the top of my head, but I can show you in the code / metrics file / by re-running evaluate on val.”

---

# Part 1 — Elevator pitch (30 seconds)

> “Our project watches a person from a webcam or video and labels what they’re doing: **standing, sitting, walking, running, or lying**. We didn’t train our own neural network. We use **pretrained YOLO** to find the person, **MediaPipe Pose** for body joints, then **hand-written geometric rules** — angles and speeds — to decide the activity. There’s a Gradio UI called MotionSense for video, photo, and live camera. On the held-out test set we got about **macro-F1 0.772** and around **15 FPS** on a laptop CPU.”

---

# Part 2 — What we use (say this when they ask “what tools / models?”)

## Software & libraries

| What | Why we use it (student wording) |
|---|---|
| **Python** | Whole pipeline and UI |
| **OpenCV** | Read frames, resize, CLAHE, MOG2 background subtraction, drawing |
| **Ultralytics YOLO11n** | Finds *where* the person is (bounding box). Nano = small/fast for CPU |
| **MediaPipe Pose (lite)** | 33 body landmarks (shoulders, hips, knees, ankles…) |
| **NumPy** | Angles, ratios, speeds from landmarks |
| **Gradio** | Browser UI so reviewers don’t need CLI |
| **scikit-learn / matplotlib** | Metrics, confusion matrix for the report |
| **imageio-ffmpeg** | Write annotated MP4 that plays in the browser |

## Datasets

| Dataset | What it is | Role |
|---|---|---|
| **GMDCSA24** | Fixed camera in a room, day/night, 4 subjects | Closest to our use case (home/room monitor) |
| **HMDB51** | Movie/web clips of walk & run | Stress test: moving camera, harder |

We split **by subject or by movie** so the same person never appears in both train and test. We **tune only on validation**, and report the final number on **test** once.

## What we did *not* use / train

- We did **not** train YOLO or MediaPipe — they’re pretrained.  
- We did **not** train a deep action classifier (too little data, harder to explain, slower).  
- We **tune thresholds** (knee angle, hip speed, etc.) on val with a random search.  
- **Fall detection** from the proposal is **not built** — we can label *lying*, not “sudden fall event.”

---

# Part 3 — How it works (walk them through like a student)

Think of **one video frame** going through four stages. Same path in the app and in evaluation.

### Stage 1 — Preprocess (`src/preprocess.py`)

1. Resize so the long side is **640 px** (YOLO was trained around that size; keeps us real-time).  
2. **CLAHE** on the lightness channel — helps in dim rooms without wrecking colors.  
3. A blurred grayscale copy is only for **MOG2** motion; YOLO and pose get the sharp color image.

> “If the room is a bit dark, CLAHE boosts local contrast so joints are easier to find.”

### Stage 2 — Infer (`src/pipeline.py` + detector / pose / motion)

1. **YOLO** looks for people (`conf = 0.4`). If several people appear, we take the **largest box** — usually the closest person. We only classify that one.  
2. **MediaPipe** estimates the skeleton (33 points). If the person is small or pose misses, we run pose again on an **upscaled crop** of the YOLO box.  
3. **Camera motion:** we estimate how the background moved (optical flow + RANSAC) so a pan doesn’t look like the person is running.  
4. **MOG2** (baseline only): moving blobs without pose — used to compare against our main method.

> “YOLO says *someone is here*. Pose says *how their body is shaped*. Rules say *what activity that shape and motion means*.”

### Stage 3 — Postprocess / classify (`src/features.py` + `src/classifier.py`)

From the landmarks we compute **scale-free** features (work near or far):

- Knee / hip **angles** (side-view sitting)  
- **Thigh ratio** (front-view sitting — thighs point at the camera)  
- **Torso tilt** (lying if nearly horizontal)  
- **Hip / leg speed** in body-heights per second (+ stride check so fidgeting ≠ walking)

**Rule order (important to say):**

1. No usable hips / legs → **unknown** (e.g. desk webcam, only face)  
2. Strong torso tilt → **lying**  
3. Folded legs / small thigh ratio while hips still → **sitting**  
4. Fast hip/leg + stride → **running**, else moving → **walking**, else → **standing**

Then a short **majority vote + persistence** so the label doesn’t flicker every frame.

### Stage 4 — Render (`src/visualize.py`)

Draw person box, skeleton, activity label, confidence, FPS. That’s what you see in Gradio.

### App extras (`app.py` + `ui.py`)

- Models load **once** at startup (reloading every frame would kill FPS).  
- Tabs: Overview, Video Analysis, Photo Analysis, Live Camera, About.  
- Live: Start camera → allow browser → orange record if needed → Stop turns tracks off.

---

# Part 4 — Results you should remember

| Fact | Number / answer |
|---|---|
| Main metric | **macro-F1 0.772** (test, tag v1.0) |
| Accuracy | **0.756** |
| MOG2 baseline macro-F1 | **0.081** (why pose+rules matter) |
| Speed | ~**66 ms/frame** → **~15.1 FPS** (eval laptop CPU) |
| Fixed room camera | ~**93%** |
| Movie / moving camera | ~**58%** |
| Why macro-F1? | Classes uneven — only **3** lying test segments |
| Tune on | **val only**, seed **106** |
| Reproduce | `python src/evaluate.py --split test` (uses `outputs/cache/`) |

> “Accuracy alone can hide bad rare classes. Macro-F1 averages each class’s F1 equally, so lying and running still count.”

---

# Part 5 — Demo tips (what to say while showing)

**Safe Analyze order:** sitting → walking → standing → lying → running (`tests/demo_tomorrow/` or `tests/cases/`).

**Live camera:** stand **2–5 m** away, **full body**, preferably side / ¾ view, **fixed** laptop.  
Desk face-only → **UNKNOWN – full body needed** (by design, not a crash).

**Honest fail line:**  
“This one fails because of [darkness / arm exercise / moving camera]. We documented it in the error analysis — we’re not claiming 100%.”

---

# Part 6 — Q&A (answer like a student)

Practice out loud. Keep answers under ~30–45 seconds unless they dig deeper.

---

### Basics / scope

**Q: What does your system do?**  
**A:** It finds a person in a video or webcam and classifies five everyday activities: standing, sitting, walking, running, lying. It’s for monitoring-style use with a fixed camera, not a medical device.

**Q: Did you train a model?**  
**A:** No. YOLO and MediaPipe are pretrained. We wrote geometric rules and tuned their thresholds on the validation set only.

**Q: Why not deep learning for the activity itself?**  
**A:** We don’t have thousands of labeled frames, we need something explainable for defense, and it has to run on a normal laptop CPU. Rules on angles and speeds fit that better here.

**Q: Does it detect falls?**  
**A:** Not yet. Lying is one class, but we didn’t implement “sudden drop = fall.” That’s future work from the proposal.

**Q: Can it handle multiple people?**  
**A:** YOLO can detect several boxes, but we only classify the **largest** person — the primary subject. Others may get a gray box with no activity. Multi-person tracking is future work.

---

### Pipeline / “how”

**Q: Walk me through one frame.**  
**A:** Resize and CLAHE → YOLO person box → MediaPipe landmarks → compute angles and speeds → rules pick a label → temporal smoothing → draw overlay. Same pipeline in the app and evaluate.

**Q: What is YOLO doing here?**  
**A:** Object detection for the COCO *person* class. It gives a box and confidence. We use conf 0.4 and take the largest box as our subject.

**Q: What is MediaPipe Pose?**  
**A:** A pose estimator that returns 33 body landmarks with visibility scores. We ignore low-visibility joints so we don’t trust guessed off-screen points.

**Q: What is MOG2 / your baseline?**  
**A:** Background subtraction — pixels that don’t look like the learned background. Our baseline classifies the big moving blob by shape and speed, with no pose. It’s weak when someone sits still (they fade into the background). Main method is much stronger: F1 0.772 vs about 0.081.

**Q: Why resize to 640?**  
**A:** Matches how YOLO was trained and keeps processing fast enough. Bigger input mostly costs time without helping us much.

**Q: Why CLAHE?**  
**A:** Dim rooms flatten the image. CLAHE boosts contrast locally so pose and detection work better at night-with-lights, without the side effects of global equalization.

**Q: Why angles and body-heights/s instead of raw pixels?**  
**A:** They’re scale-free — same thresholds if you’re 2 m or 4 m away, and clothing doesn’t change joint geometry as much as appearance does.

---

### Classification details

**Q: How do you tell sitting from standing?**  
**A:** Side view: bent knee and hip angles. Front view: thigh ratio is small because the thigh points toward the camera. We also require the hips not to be traveling, so a walker isn’t called sitting.

**Q: How do you tell walking from running?**  
**A:** Higher hip/leg speed thresholds for running, plus a stride check so both feet jiggling in place doesn’t count as locomotion.

**Q: Why did my desk webcam say unknown / no person?**  
**A:** Only the face/upper body is visible. Without hips and legs we refuse to guess — MediaPipe used to invent fake poses (even “lying”) on close-ups. For live demo we step back so the full body is in frame.

**Q: What if only a hand is in front of the lens?**  
**A:** We added a pose-plausibility gate: a real body needs shoulders, hips, and a leg joint, and the skeleton must match the YOLO box. Hand fakes get rejected.

---

### Data / evaluation

**Q: What data did you use?**  
**A:** GMDCSA24 room videos and HMDB51 walk/run clips. Test is held-out subjects/movies. We also have short `tests/cases` and `tests/demo_tomorrow` clips for the UI.

**Q: Why is movie accuracy lower?**  
**A:** HMDB often has a moving camera, cuts, and odd framing. Our design assumes a mostly fixed camera. Room footage is around 93%; movies around 58%.

**Q: Why validation and test?**  
**A:** Val is for choosing thresholds and design changes. Test is frozen for the final reported score so we don’t accidentally overfit the number we publish.

**Q: What is overfitting in your project?**  
**A:** Thresholds that fit quirks of val (or one dataset) and fail on new people/movies. We even rejected a retune that looked great on val but worse on test.

**Q: What does the confusion matrix show?**  
**A:** Sitting became strong after M4 fixes. Running is weakest, especially on movies. Slow walking sometimes gets absorbed into standing — a known limit.

**Q: How do you reproduce the numbers without downloading 3 GB of video?**  
**A:** Cached per-frame features under `outputs/cache/`. `evaluate.py` replays the rules on those traces.

---

### Metrics / CV concepts (reviewer favorites)

**Q: Why macro-F1 not accuracy?**  
**A:** Classes are imbalanced. Accuracy can look okay while rare classes like lying are ignored. Macro-F1 averages per-class F1 equally.

**Q: What is IoU? Did you use it?**  
**A:** Intersection over union of two boxes. We didn’t train a detector or score activity with IoU. YOLO’s threshold is **confidence 0.4**. We score activity by majority label over a time segment.

**Q: What is NMS?**  
**A:** Non-maximum suppression — when many overlapping boxes hit the same person, keep the best one and suppress the rest. YOLO does that inside; we then pick the largest remaining person.

**Q: What is optical flow? RANSAC?**  
**A:** Optical flow estimates pixel motion between frames. We use background flow to estimate camera motion. RANSAC fits that model while ignoring outlier points, better than a plain average when some matches are wrong.

**Q: Convolution / CNN — do you use that?**  
**A:** YOLO’s backbone is a CNN — early layers find edges/textures, deeper layers find object parts. We didn’t train those weights; we only consume person boxes.

---

### “What if we change X?” (quick predictions)

| They ask | You say |
|---|---|
| Raise YOLO conf to 0.7? | More misses → more “no person,” especially dark/lying |
| Lower YOLO conf to 0.2? | More false boxes on chairs/coats; wrong primary person |
| Remove CLAHE? | Darker rooms get worse landmarks |
| Bigger blur for everything? | Hurts YOLO/pose (edges); blur is only for MOG2 |
| Tune on test? | Dishonest / overfit the report number |
| Run YOLO every frame? | Slightly stabler boxes, slower FPS |
| Twice the distance? | Person too small → pose/YOLO weak |

---

### App / engineering

**Q: Why is live FPS lower than 15?**  
**A:** Eval measures processing of stored video. Live also depends on browser capture, Gradio streaming (~4 Hz update), and sharing the CPU with the UI. Still usable for demo if framing is right.

**Q: Why load the model once?**  
**A:** Loading YOLO every frame would drop us to about 1 FPS. We warm up once at startup and reuse the detector.

**Q: Gradio / browser camera still on after Stop?**  
**A:** We call Stop which clears our pipeline state and runs JS to stop `getUserMedia` tracks so the webcam LED turns off.

---

### Limitations & future (be honest)

**Q: When does it fail?**  
**A:** Moving / handheld cameras, near-total darkness, very small people, arm exercise while standing (looks like walking), close-ups without legs, multi-person scenes beyond the largest person.

**Q: What would you add next?**  
**A:** Our own webcam dataset, more lying examples, real fall detection (fast torso drop), maybe multi-person IDs — still keeping the fixed-camera setting.

**Q: Is this ready for hospitals / real security?**  
**A:** No — it’s a course prototype. Good for a fixed, well-lit room demo with full body in view. Not a medical or safety-certified system.

---

### Team / ownership (if asked)

**Q: Who did what?**  
**A:** Roughly — Wilmark: pipeline rules, evaluation, app integration, report numbers. Porcopio: detection/pose/camera-motion path, data import/splits, test cases, docs/slides support. Both own the final demo and Q&A. (See contribution statements in `docs/m6/`.)

---

# Part 7 — 2-minute oral checklist

1. Pitch (Part 1)  
2. “We use YOLO + MediaPipe + rules; no training”  
3. Four stages in one sentence each  
4. One number: macro-F1 **0.772**, room **93%**, movie **58%**  
5. One limitation + one future work  
6. Offer to open `src/classifier.py` or run a sample clip  

---

# Quick file map (if they open the IDE)

| File | One-line student gloss |
|---|---|
| `src/pipeline.py` | Glue: frame → all stages |
| `src/detector.py` | YOLO person boxes |
| `src/pose.py` | MediaPipe landmarks + crop fallback |
| `src/features.py` | Angles, thigh ratio, speeds, pose gate |
| `src/classifier.py` | Rules + temporal filter |
| `src/evaluate.py` | Score val/test, confusion matrix |
| `configs/tuned_params.json` | Thresholds chosen on val |
| `app.py` / `ui.py` | MotionSense Gradio UI |
| `tests/cases/` | Built-in sample clips |
| `tests/demo_tomorrow/` | Extra Analyze pack for presentation |

---

*Also see:* `docs/m6/QA_Reviewer_Pack.md` (shorter numeric pack), `docs/Presentation_Script.md` (talk track), `docs/design_choices.md` (deeper “why”).
