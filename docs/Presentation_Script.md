# Final Presentation Script (≈10 minutes + demo + Q&A)

**Team:** Porcopio Zheniel Ivan · Veridiano Wilmark  
**Slides:** `docs/slides_final.pptx` · **App:** `python app.py` → http://127.0.0.1:7860  
**Backup:** `docs/backup_demo_m4.mp4`

---

## Who does what

| Segment | Time | Speaker | Slides |
|---|---|---|---|
| Title + problem & objectives | ~1 min | **Wilmark** | 1–2 |
| Dataset + approach + pipeline | ~2 min | **Porcopio** | 3–5 |
| Results + sample outputs | ~2 min | **Wilmark** | 6–7 |
| Live demo | ~4 min | **Porcopio drives**, Wilmark narrates | 8 |
| Limitations + future work | ~1 min | **Porcopio** | 9–10 |
| Thank you / Q&A | — | **Both** | 11 |

---

## Script (say this, don’t read slides word-for-word)

### 1. Title (Wilmark, 15 s)
“Good day. We’re Porcopio and Veridiano. Our project is a **real-time activity monitor**: from a laptop webcam or video, we find a person and label **standing, sitting, walking, running, or lying** — using YOLO, MediaPipe Pose, and geometric rules. Nothing trained by us. Tag **v1.0**.”

### 2. Problem & objectives (Wilmark, 45 s)
“Manual watching is tiring and slow. Cheap motion alarms don’t say *what* the person is doing.  
We built: live + recorded input; person detect and track; five activity classes; a Gradio UI with boxes and labels; and evaluation for accuracy, FPS, and lighting.  
**Falls / sudden abnormal motion** from the proposal are **future work** — we detect lying, not a fall event yet.”

### 3. Dataset (Porcopio, 45 s)
“Two public sources. **GMDCSA24**: fixed room camera, day and night. **HMDB51**: movie clips, often a moving camera.  
We split **by subject or movie**, so the same person never leaks into test. Tune on **val only**; report on **90 held-out test segments**. That honesty is why room scores high and movies score lower.”

### 4. Approach (Porcopio, 45 s)
“Four stages: **preprocess** (resize 640, CLAHE), **infer** (YOLO person, MediaPipe 33 landmarks, camera motion with RANSAC, MOG2 for baseline), **postprocess** (angles, thigh ratio, hip speed minus camera, rules), **render** (boxes, skeleton, label, FPS).”

### 5. Pipeline (Porcopio, 30 s)
“Point at the diagram. Same pipeline in `app.py` and `evaluate.py`. Models load once.”

### 6. Results (Wilmark, 90 s)
“On test: **macro-F1 0.772**, accuracy **0.756**, versus MOG2 baseline **0.081**.  
**15.1 FPS** on our laptop CPU after warm-up.  
Fixed room camera: **93%** day and night. Movie clips: **58%**. Sitting recall went from **0.39 to 1.00** after Milestone 4 fixes.  
We use **macro-F1** because classes are imbalanced — only 3 lying segments.”

### 7. Samples (Wilmark, 30 s)
“Show confusion matrix and one pass + one fail. We don’t hide failures.”

### 8. Live demo (Porcopio drives · Wilmark talks, 4 min)

**Order:**
1. Open Gradio → **Video** tab → run **TC01 sit** or **TC03 walk** (built-in examples).  
2. One **hard pass**: TC05 night sit.  
3. One **honest fail**: TC08 arm exercise or TC09 dark — “limitation, documented.”  
4. **Webcam**: stand **2–3 m**, **side view**, full body — stand / sit / walk.  
5. Click **Stop camera** when done.  
6. If live fails → play `backup_demo_m4.mp4`.

**Wilmark line:** “Operating conditions: fixed camera, whole body, room light. Close-ups and feet-toward-camera lying are outside that.”

### 9. Limitations & future (Porcopio, 60 s)
“Not reliable for: moving cameras, near-darkness, tiny people, arm exercise while standing.  
Future: own webcam dataset, more lying data, **fall detection** (fast torso drop → lying).”

### 10. Close (both, 15 s)
“Thank you. Questions?”

---

## Demo checklist (before you stand up)

- [ ] `python app.py` already running; models warmed up  
- [ ] Browser on http://127.0.0.1:7860  
- [ ] Test clips load; room lighting OK  
- [ ] Backup video on USB / Desktop  
- [ ] Slides open; projector tested  
- [ ] Close Extra Chrome tabs / Discord (CPU)

---

## One-liners for hard questions

| Question | Answer |
|---|---|
| Why not 30 FPS? | YOLO+pose on CPU ≈ 66 ms/frame; Gradio webcam ~7–10 FPS. Still usable. |
| Did you train? | No — pretrained YOLO/MediaPipe; we tune **rules on val**. |
| Why macro-F1? | Class imbalance (lying=3, walking=36). |
| Falls? | Future work; lying yes, fall event no. |
| IoU? | Detector concept; we use YOLO **conf 0.4**, score **activity segments**, not box IoU. |
