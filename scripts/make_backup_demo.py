"""Build a short backup demo reel from annotated Milestone 4 test cases.

    python scripts/make_backup_demo.py

Writes docs/backup_demo_m4.mp4 (~under 5 minutes): title cards + annotated
clips for passes and known hard failures. For the Gradio + live webcam part,
screen-record python app.py separately and keep that file next to this one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "outputs" / "test_cases"
OUT = ROOT / "docs" / "backup_demo_m4.mp4"
W, H, FPS = 960, 540, 15

# Keep the reel short: a few passes + hard fails the guide wants shown honestly.
CLIPS = [
    ("TC01 sitting ✔", "TC01_tc01_sit_floor_eating.mp4", 6),
    ("TC02 standing ✔", "TC02_tc02_stand_window.mp4", 5),
    ("TC03 walking ✔", "TC03_tc03_walk_room.mp4", 5),
    ("TC05 night sit (hard) ✔", "TC05_tc05_sit_talk_night.mp4", 5),
    ("TC07 front sit (hard) ✔", "TC07_tc07_sit_bed_edge_front.mp4", 5),
    ("TC08 arm exercise (hard) ✘", "TC08_tc08_stand_arm_exercise.mp4", 5),
    ("TC09 very dark (hard) ✘", "TC09_tc09_walk_very_dark.mp4", 4),
    ("TC11 portrait ✔", "TC11_tc11_stand_portrait.mp4", 4),
]


def title_frame(text: str, sub: str = "") -> np.ndarray:
    img = np.full((H, W, 3), (26, 48, 44), np.uint8)
    cv2.putText(img, "Activity Monitor — Milestone 4 backup", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (61, 220, 151), 2, cv2.LINE_AA)
    y0 = 220
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (40, y0 + i * 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (244, 247, 245), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (40, H - 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (168, 189, 182), 1, cv2.LINE_AA)
    return img


def fit(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), (15, 28, 26), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def write_still(writer, frame: np.ndarray, seconds: float) -> None:
    n = max(1, int(seconds * FPS))
    for _ in range(n):
        writer.write(frame)


def main() -> int:
    missing = [name for _, name, _ in CLIPS if not (CASES / name).exists()]
    if missing:
        print("Missing annotated clips in outputs/test_cases/:", ", ".join(missing))
        print("Run: python scripts/run_test_cases.py")
        return 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))
    if not writer.isOpened():
        print("Could not open VideoWriter for", OUT)
        return 1

    write_still(writer, title_frame("Backup demo reel\nAnnotated test cases",
                                    "App: python app.py  ·  Live webcam in class"), 3.0)

    for label, name, max_sec in CLIPS:
        write_still(writer, title_frame(label, name), 1.2)
        cap = cv2.VideoCapture(str(CASES / name))
        n_max = int(max_sec * FPS)
        i = 0
        while i < n_max:
            ok, frame = cap.read()
            if not ok:
                break
            out = fit(frame)
            cv2.putText(out, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (61, 220, 151), 2, cv2.LINE_AA)
            writer.write(out)
            i += 1
        cap.release()

    write_still(writer, title_frame("End of reel\n9/13 file cases pass",
                                    "Hard fails shown on purpose · TC14/15 live"), 2.5)
    writer.release()
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
