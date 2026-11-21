# Extension: Fall / Sudden-Drop Alert

**Why this (not face attendance):** Other groups built face recognition for attendance. Our project is an **activity monitor**. Copying attendance would look bolted on. Fall alert completes **Objective 6** from our proposal (“sudden abnormal movement”) that was still “future work.”

## What we added

| Piece | Role |
|---|---|
| `src/fall_alert.py` | Heuristic: upright → fast hip drop / torso tilt → lying |
| `src/pipeline.py` | Runs the monitor every frame; attaches `fall_alert`, `fall_events` |
| `src/visualize.py` | Red **FALL ALERT** banner on the annotated frame |
| Gradio video / live | Headline shows fall count / live “FALL ALERT” |

**Not a sixth class.** Evaluate still scores standing/sitting/walking/running/lying. The alert is an **event overlay** for monitoring demos.

## How it works (say this)

> “We already know posture from pose rules. The extension keeps a short history of hip height and torso tilt. If the person was upright and then, within about a second, the hips drop fast or the torso tilts hard into a lying pose, we raise a **FALL ALERT** for a couple of seconds and count the event.”

## Demo tip (Tuesday)

1. Video: any GMDCSA **Fall** clip / room fall segment — watch for red banner + `fall alerts` in details.  
2. Live: full body, 2–5 m; carefully drop from stand/sit to lie on a mat (safe demo).  
3. Honest line: “Heuristic extension for the proposal’s abnormal-motion goal — not a certified medical fall detector.”

## Q&A

**Q: Why not face recognition like other groups?**  
**A:** Different problem. Ours is activity + safety monitoring with a fixed camera. Face attendance needs identity data and a different pipeline; fall alert reuses our pose features.

**Q: Does this change your 0.772 macro-F1?**  
**A:** No. Fall alert doesn’t change the five activity labels used in evaluate.

**Q: False alarms?**  
**A:** Fast sit-downs or exercise can trigger. We use a cooldown and require an upright→down pattern with speed/tilt thresholds.
