# Test cases

Written before the run (inputs, conditions and expected labels in `tests/test_cases.csv`), then run with `python scripts/run_test_cases.py` on 2026-11-12. Annotated outputs: `outputs/test_cases/`.

**9 of 13 passed** (4 of 7 hard cases). Confidence = share of frames that agree with the reported label. Speed = median ms per frame after 10 warm-up frames.

| ID | Input | Condition | Expected | Actual | Conf. | ms/frame | Pass |
|---|---|---|---|---|---:|---:|:-:|
| TC01 | `cases/tc01_sit_floor_eating.mp4` | Day, sitting on the floor eating, side view | sitting | sitting | 0.87 | 51 | ✔ |
| TC02 | `cases/tc02_stand_window.mp4` | Day, standing at the window opening the curtain | standing | standing | 0.94 | 57 | ✔ |
| TC03 | `cases/tc03_walk_room.mp4` | Day, walking across the room | walking | walking | 0.89 | 61 | ✔ |
| TC04 | `cases/tc04_lie_floor.mp4` | Day, lying down on the floor | lying | no person | 0.78 | 61 | ✘ |
| TC05 | `cases/tc05_sit_talk_night.mp4` | HARD: night, seated on the bed talking on the phone (M3 failure T026: limb motion read as walking) | sitting | sitting | 0.99 | 63 | ✔ |
| TC06 | `cases/tc06_sit_bed_drink.mp4` | HARD: day, seated on the bed drinking (M3 failure T027: limb motion read as walking) | sitting | sitting | 0.89 | 52 | ✔ |
| TC07 | `cases/tc07_sit_bed_edge_front.mp4` | HARD: night, sitting on the bed edge facing the camera (M3 failure T042: looks like standing in 2D) | sitting | sitting | 0.91 | 52 | ✔ |
| TC08 | `cases/tc08_stand_arm_exercise.mp4` | HARD: standing while exercising the arms and bending (M3 failure T009) | standing | walking | 0.44 | 54 | ✘ |
| TC09 | `cases/tc09_walk_very_dark.mp4` | HARD: night clip darkened (gamma 2.2 and x0.35 brightness), walking | walking | no person | 0.50 | 60 | ✘ |
| TC10 | `cases/tc10_sit_half_res.mp4` | HARD: half resolution (320x180), sitting on the floor at night | sitting | no person | 0.69 | 57 | ✘ |
| TC11 | `cases/tc11_stand_portrait.mp4` | HARD: portrait 9:16 frame like a phone video, standing | standing | standing | 0.99 | 55 | ✔ |
| TC12 | `cases/tc12_sit_photo.jpg` | Edge: a single photo instead of a video, sitting on the bed | sitting | sitting | 1.00 | 134 | ✔ |
| TC13 | `cases/tc13_empty_room.jpg` | Edge: photo of the room with nobody in it | no person | no person | 1.00 | 171 | ✔ |
| TC14 | `webcam` | Live: laptop webcam, person seated at the desk, only the upper body visible | unknown | (run live: --webcam 0) |  |  | – |
| TC15 | `webcam` | Live: laptop webcam, person 2-3 m away walking across the room, full body | walking | (run live: --webcam 0) |  |  | – |
