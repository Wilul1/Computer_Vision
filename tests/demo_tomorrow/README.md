# Demo pack for Analyze (tomorrow)

14 short clips, mostly **GMDCSA24 Subject 4** (different person from `tests/cases`, which used Subject 3), plus HMDB run/walk.

Folder: `tests/demo_tomorrow/`  
Labels: `manifest.csv`

## Safe order for live Analyze

| # | File | Expect | Smoke result |
|---|------|--------|--------------|
| 1 | `d01_sit_chair_day.mp4` | sitting | sitting |
| 2 | `d03_walk_pick_book_day.mp4` | walking | walking |
| 3 | `d05_stand_with_jar_day.mp4` | standing | standing |
| 4 | `d07_lie_bed_day.mp4` | lying | lying |
| 5 | `d11_run_longestyard.mp4` | running | running (use this for run) |
| 6 | `d04_walk_phone_dark.mp4` | walking | walking (dark) |

## Skip or explain if asked

- `d12_run_fugitive.mp4` — often reads as standing/walking (movie framing). Prefer **d11**.
- `d14_walk_movie_eurotrip.mp4` — moving camera; harder.
- Short stands (`d10`) can flicker; use **d05** / **d06**.

## How to open in the app

1. `python app.py`
2. Video Analysis tab → upload a file from `tests/demo_tomorrow/`
3. Or browse that folder in the file picker

These files are local only (not required in git).
