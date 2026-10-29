"""Copy clips from a public dataset into data/raw/<activity>/ with a SOURCE id, so
make_splits.py keeps each original video/person in one split.

    python scripts/import_public.py hmdb51 D:/datasets/hmdb51      # folder with walk/, run/, ...
    python scripts/import_public.py kth    D:/datasets/kth         # folder with person01_walking_d1_uncomp.avi ...

HMDB51 (Brown University, non-commercial research use)
  File names: <movie>_<action>_<body>_<camera>_<people>_<view>_<quality>_<n>.avi
  We keep only clips where the FULL body is visible (body == "f") and ONE person is
  in the shot (people == "np1"); bad-quality clips are skipped. Source = movie name.
  HMDB "sit" and "stand" are the *transitions* (sitting down / standing up), not
  postures, so they are only imported with --with-transitions.

KTH (KTH Royal Institute of Technology, non-commercial research use)
  File names: person<NN>_<action>_d<scenario>_uncomp.avi. Source = person, so the
  same subject never appears in two splits. "jogging" is skipped (between our
  walking and running classes). Frames are 160x120, so they are upscaled 4x
  (MediaPipe struggles with people only ~80 px tall).
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

RAW = config.ROOT / "data" / "raw"
HMDB_MAP = {"walk": "walking", "run": "running"}
HMDB_TRANSITIONS = {"sit": "sitting", "stand": "standing"}
KTH_MAP = {"walking": "walking", "running": "running"}


def clean(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in s).strip("-").lower()[:40]


def import_hmdb51(root: Path, with_transitions: bool, limit: int):
    mapping = HMDB_MAP | (HMDB_TRANSITIONS if with_transitions else {})
    copied = skipped = 0
    for action, activity in mapping.items():
        clips = sorted((root / action).glob("*.avi"))
        if not clips:
            print(f"  no clips in {root / action} (extract {action}.rar there)")
        per_source = {}
        for clip in clips:
            tokens = clip.stem.split("_")
            # From the right: action, body, camera, people, view, quality, n
            body, people, quality = tokens[-6], tokens[-4], tokens[-2]
            if body != "f" or people != "np1" or quality == "bad":
                skipped += 1
                continue
            movie = clip.stem[: clip.stem.rfind(f"_{action}_")] or "unknown"
            source = f"hmdb-{clean(movie)}"
            per_source[source] = per_source.get(source, 0) + 1
            if limit and per_source[source] > limit:
                skipped += 1
                continue
            dest = RAW / activity / f"{source}__{per_source[source]:02d}.avi"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip, dest)
            copied += 1
    print(f"HMDB51: copied {copied}, skipped {skipped} (not full body / several people / bad quality / over limit)")


def upscale_copy(src: Path, dest: Path, factor: int):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
    cap.release()
    if writer:
        writer.release()


def import_kth(root: Path, limit: int):
    copied = 0
    per_source = {}
    for clip in sorted(root.rglob("person*_*_d*_uncomp.avi")):
        person, action, scenario = clip.stem.split("_")[:3]
        if action not in KTH_MAP:
            continue
        source = f"kth-{person}"
        per_source[source] = per_source.get(source, 0) + 1
        if limit and per_source[source] > limit:
            continue
        dest = RAW / KTH_MAP[action] / f"{source}__{scenario}-{per_source[source]:02d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        upscale_copy(clip, dest, 4)
        copied += 1
    print(f"KTH: copied {copied} clips (upscaled 4x)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["hmdb51", "kth"])
    p.add_argument("folder", type=Path)
    p.add_argument("--with-transitions", action="store_true", help="HMDB51: also import sit/stand")
    p.add_argument("--limit-per-source", type=int, default=0, help="max clips per movie/person (0 = all)")
    args = p.parse_args()
    if not args.folder.exists():
        sys.exit(f"Not found: {args.folder}")
    if args.dataset == "hmdb51":
        import_hmdb51(args.folder, args.with_transitions, args.limit_per_source)
    else:
        import_kth(args.folder, args.limit_per_source)
    print("Next: python scripts/make_splits.py")


if __name__ == "__main__":
    main()
