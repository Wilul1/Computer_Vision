# Individual Contribution Statement — Milestone 6

**Veridiano Wilmark** · CSST 106 Computer Vision · Laguna State Polytechnic University  
Team: Porcopio Zheniel Ivan · Veridiano Wilmark  
Repository: https://github.com/Wilul1/Computer_Vision · tag `v1.0`

**Note on git history.** All pushes used one GitHub account (`Wilmark Veridiano`). Pair work on shared modules is claimed below and matches Appendix C of the final report. Commit links still identify the milestones where my parts landed.

## 1. What I built

| Area | Paths |
|---|---|
| Pipeline orchestration | `src/pipeline.py`, `src/run.py`, `src/config.py` |
| Features + rule classifier + params | `src/features.py`, `src/classifier.py`, `src/params.py` |
| Evaluation & reproducibility | `src/evaluate.py`, `outputs/cache/`, `configs/tuned_params.json` |
| Tuning | `scripts/tune.py` |
| Gradio app | `app.py` |
| Preprocess / MOG2 baseline support | `src/preprocess.py`, `src/motion.py` (shared) |

**Key commits (3–5):**

1. [725543a](https://github.com/Wilul1/Computer_Vision/commit/725543a) — Milestone 2 prototype (pipeline + first rules)  
2. [b4a0e97](https://github.com/Wilul1/Computer_Vision/commit/b4a0e97) — Milestone 3: `evaluate.py`, baseline comparison, tuning  
3. [cf19105](https://github.com/Wilul1/Computer_Vision/commit/cf19105) — Milestone 4: Gradio app, M3-error fixes, reproducible test numbers  
4. [6863b05](https://github.com/Wilul1/Computer_Vision/commit/6863b05) — Milestone 5 package wiring (README reproduce path, final deliverables)  
5. [7fc9eb2](https://github.com/Wilul1/Computer_Vision/commit/7fc9eb2) — One-command evaluation + design-choices doc  

## 2. A decision I made

**Choice:** score tuning with the **mean macro-F1 of each dataset** (GMDCSA24 and HMDB51 equally), not overall macro-F1 on all val segments pooled.

**Rejected:** overall macro-F1 on the pooled val set (what Milestone 3 used).

**Evidence:** pooled F1 let ~55 movie walking segments outvote room sitting segments, so sitting recall on test collapsed to 0.39 while movie scores rose. Re-tuning with equal dataset weight (plus the M4 feature options) moved val balanced F1 **0.595 → 0.637**, and sitting recall on test later reached **1.00**. Documented in `docs/limitations.md` and `docs/final_report.md` §6. Seed for search: **106** (`scripts/tune.py`).

## 3. A problem I solved

**Problem:** seated people moving their arms/legs were labelled **walking** (11 of 31 Milestone 3 test failures: “limb motion in place”).

**Cause:** speed / leg-swing rules could fire before posture was checked; jitter of both ankles looked like a step.

**Fix:** (1) check folded-leg / sitting posture **before** speed rules; (2) require ankles to move in **opposite** directions for a stride; (3) retune with the balanced objective above. Result: hard test cases TC05–TC07 pass; sitting recall 0.39 → 1.00 on the locked test set. Trade-off documented: the YOLO person-box gate that removed false skeletons (TC13) also hides some lying/dark people (TC04, TC09).

## 4. What I learned

1. **Validation vs test / overfitting of thresholds** — with ~20 thresholds and ~100 val segments, random search is noisy; a second search scored 0.650 on val but worse on test and was rejected (`configs/tuned_params_retune_rejected.json`). Val is for choosing; test is reported once.  
2. **Why a baseline matters** — the MOG2-blob baseline (macro-F1 **0.081**) shows what pose geometry adds (main **0.772**) on the **same** frames, not just “the model works.”  

**One thing I would do differently:** record our own fixed-camera webcam clips earlier and tune with a third “own webcam” condition, so operating claims are measured on the real demo setup—not only public room/movie data. Fall / sudden-abnormal detection would be the next objective after that.
