"""Cut the test-case clips listed in tests/test_cases.csv from GMDCSA24 (MIT
licence, so they may live in the repo). Run once; the outputs are committed.

    python scripts/make_test_cases.py

Every clip is resized to 640 px wide (the pipeline works at <= 640 anyway) and
written as H.264 to keep the repo small. The hard cases are made here:
  TC09  darkened: gamma 2.2 and x0.35 (the "night" clips are lit by room lights)
  TC10  half resolution, 320x180
  TC11  cropped to a 9:16 portrait frame around the person, like a phone video
"""
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "test" / "_segments"
OUT = ROOT / "tests" / "cases"

# id: (source video, start s, end s, transform)
CLIPS = {
    "tc01_sit_floor_eating.mp4": ("gmdcsa-s3__adl-16.mp4", 0, 10, None),
    "tc02_stand_window.mp4": ("gmdcsa-s3__adl-10.mp4", 0, 12, None),
    "tc03_walk_room.mp4": ("gmdcsa-s3__fall-14.mp4", 0, 4.6, None),
    "tc04_lie_floor.mp4": ("gmdcsa-s3__adl-08.mp4", 0.8, 4, None),
    "tc05_sit_talk_night.mp4": ("gmdcsa-s3__adl-20.mp4", 0, 9, None),
    "tc06_sit_bed_drink.mp4": ("gmdcsa-s3__adl-21.mp4", 0, 10, None),
    "tc07_sit_bed_edge_front.mp4": ("gmdcsa-s3__fall-18.mp4", 0, 3.2, None),
    "tc08_stand_arm_exercise.mp4": ("gmdcsa-s3__adl-05.mp4", 0, 5, None),
    "tc09_walk_very_dark.mp4": ("gmdcsa-s3__adl-19.mp4", 1.8, 5, "dark"),
    "tc10_sit_half_res.mp4": ("gmdcsa-s3__adl-13.mp4", 0, 8, "half"),
    "tc11_stand_portrait.mp4": ("gmdcsa-s3__adl-02.mp4", 0, 9, "portrait"),
}
PHOTOS = {
    "tc12_sit_photo.jpg": ("gmdcsa-s3__adl-22.mp4", 4.0, None),
    "tc13_empty_room.jpg": ("gmdcsa-s3__adl-22.mp4", 4.0, "empty"),
}
DARK_LUT = np.array([((i / 255) ** 2.2) * 255 * 0.35 for i in range(256)], np.uint8)


def frames(name, start, end):
    cap = cv2.VideoCapture(str(SRC / name))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    out = []
    while end is None or len(out) < int((end - start) * fps):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
        if end is None:
            break
    cap.release()
    return out, fps


def portrait_box(frames_):
    """9:16 window centred on where the person is, found with the median of
    the frame-to-frame differences (the person is what moves)."""
    h, w = frames_[0].shape[:2]
    diff = np.zeros((h, w), np.float32)
    for a, b in zip(frames_[::5], frames_[5::5]):
        diff += cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    cx = int(np.argmax(diff.sum(axis=0)))
    cw = int(h * 9 / 16)
    x0 = min(max(0, cx - cw // 2), w - cw)
    return x0, cw


def transform(f, kind, box=None):
    if kind == "portrait":
        x0, cw = box
        f = f[:, x0:x0 + cw]
        return cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA)
    f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])), interpolation=cv2.INTER_AREA)
    if kind == "dark":
        f = cv2.LUT(f, DARK_LUT)
    elif kind == "half":
        f = cv2.resize(f, (320, int(f.shape[0] / 2)), interpolation=cv2.INTER_AREA)
    return f


def write_mp4(path, imgs, fps):
    h, w = imgs[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "28"], macro_block_size=1)
    writer.send(None)
    for f in imgs:
        writer.send(np.ascontiguousarray(f[:, :, ::-1]))
    writer.close()


def main():
    if not SRC.exists():
        sys.exit("Needs the GMDCSA24 test videos in data/test/_segments (python scripts/get_data.py).")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, start, end, kind) in CLIPS.items():
        imgs, fps = frames(src, start, end)
        box = portrait_box(imgs) if kind == "portrait" else None
        write_mp4(OUT / name, [transform(f, kind, box) for f in imgs], fps)
        print(f"{name}: {len(imgs)} frames, {(OUT / name).stat().st_size / 1e3:.0f} kB")
    for name, (src, at, kind) in PHOTOS.items():
        f = transform(frames(src, at, None)[0][0], None)
        if kind == "empty":
            f = f[:, int(f.shape[1] * 0.70):]   # right part of the room (door); checked by eye that nobody is in it
        cv2.imwrite(str(OUT / name), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name}: {f.shape[1]}x{f.shape[0]}")


if __name__ == "__main__":
    main()
