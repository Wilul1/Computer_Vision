"""Copy clips from a public dataset into data/raw/<activity>/ with a SOURCE id, so
make_splits.py keeps each original video/person in one split.

    python scripts/import_public.py hmdb51 D:/datasets/hmdb51      # folder with walk/, run/, ...
    python scripts/import_public.py kth    D:/datasets/kth         # folder with person01_walking_d1_uncomp.avi ...
    python scripts/import_public.py gmdcsa24 D:/datasets/GMDCSA24  # folder with "Subject 1" ... "Subject 4"

GMDCSA24 (Alam et al., Data in Brief 2024, MIT licence)
  One room, fixed camera, 4 subjects, day and night lighting. Each video has
  time-segment labels (Sitting, Sleeping, Walking, Standing, Falling, ...), so
  videos go to data/raw/_segments/ and the segments to data/annotations/segments.csv.
  Source = subject, so a person never appears in two splits.

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
import csv
import re
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

RAW = config.ROOT / "data" / "raw"
ANNOT = config.ROOT / "data" / "annotations"
INFO_CSV = ANNOT / "clip_info.csv"          # per clip: dataset + recording condition (for the test log)
SEGMENTS_CSV = ANNOT / "segments.csv"       # time-segment labels for multi-activity videos
HMDB_VIEWS = {"fr": "front", "ba": "back", "le": "left", "ri": "right"}
HMDB_MAP = {"walk": "walking", "run": "running"}
# GMDCSA24 class names -> ours. "Sleeping" there is lying on the bed. Falling and
# Reading are not posture classes of ours (Reading overlaps sitting/walking).
GMDCSA_MAP = {"sitting": "sitting", "sleeping": "lying", "lying": "lying",
              "walking": "walking", "standing": "standing", "running": "running"}
HMDB_TRANSITIONS = {"sit": "sitting", "stand": "standing"}
KTH_MAP = {"walking": "walking", "running": "running"}


def clean(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in s).strip("-").lower()[:40]


def load_csv(path: Path, key: str) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def save_info(info: dict):
    INFO_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(INFO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "dataset", "condition", "original"])
        w.writeheader()
        for clip in sorted(info):
            w.writerow({"clip": clip, **{k: info[clip].get(k, "") for k in ("dataset", "condition", "original")}})


def import_hmdb51(root: Path, with_transitions: bool, limit: int, info: dict):
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
            camera, view = tokens[-5], tokens[-3]
            info[f"{dest.parent.name}/{dest.name}"] = {
                "dataset": "hmdb51",
                "condition": f"movie; {'moving camera' if camera == 'cm' else 'static camera'}; "
                             f"{HMDB_VIEWS.get(view, view)} view; {quality} quality",
                "original": clip.name,
            }
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


NAME_RE = re.compile(r"([A-Za-z][A-Za-z ()]*?)\s*\[")
RANGE_RE = re.compile(r"([\d.]+)\s*to\s*([\d.]+)")


def parse_classes(text: str) -> list[tuple[str, str, str]]:
    """'Sitting[0 to 3]; Standing[2.4 to 3.6; 6.6 to 7.9; Walking[...' ->
    [(Sitting, 0, 3), (Standing, 2.4, 3.6), (Standing, 6.6, 7.9), ...].
    Tolerates several ranges in one bracket and missing ']' (both occur in the CSVs)."""
    marks = list(NAME_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        out += [(m.group(1), a.rstrip("."), b.rstrip(".")) for a, b in RANGE_RE.findall(body)]
    return out


def import_gmdcsa24(root: Path, info: dict):
    """Each video has several activities with time ranges, e.g.
    'Sitting[0 to 3.4]; Walking[3.4 to 6]'. Videos go to data/raw/_segments/ and
    their labelled time ranges to data/annotations/segments.csv."""
    segments = {k: v for k, v in load_csv(SEGMENTS_CSV, "id").items() if not v["clip"].startswith("gmdcsa-")}
    copied = 0
    for subject_dir in sorted(root.glob("Subject*")):
        subject = re.sub(r"\D", "", subject_dir.name)
        for kind in ("ADL", "Fall"):
            table = subject_dir / f"{kind}.csv"
            if not table.exists():
                continue
            with open(table, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))[1:]
            for row in rows:
                if not row or not row[0].strip():
                    continue
                video = subject_dir / kind / row[0].strip()
                if not video.exists():
                    print(f"  missing {video}")
                    continue
                # Columns: file, length, time of recording, attire, description..., classes
                # (the description itself may contain commas, so classes = last field).
                when, attire = row[2].strip(), row[3].strip()
                description = ",".join(row[4:-1]).strip()
                dest = RAW / "_segments" / f"gmdcsa-s{subject}__{kind.lower()}-{video.stem}.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video, dest)
                info[f"{dest.parent.name}/{dest.name}"] = {"dataset": "gmdcsa24",
                                   "condition": f"room, static camera; {when}; {attire}; {description}",
                                   "original": f"{subject_dir.name}/{kind}/{video.name}"}
                for i, (name, start, end) in enumerate(parse_classes(row[-1])):
                    label = GMDCSA_MAP.get(name.strip().lower())
                    if label:
                        sid = f"{dest.stem}#{i}"
                        segments[sid] = {"id": sid, "clip": dest.name, "start": start, "end": end, "label": label}
                copied += 1
    with open(SEGMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "clip", "start", "end", "label"])
        w.writeheader()
        w.writerows(sorted(segments.values(), key=lambda s: s["id"]))
    print(f"GMDCSA24: copied {copied} videos, {sum(s['clip'].startswith('gmdcsa-') for s in segments.values())} "
          f"labelled segments -> {SEGMENTS_CSV.relative_to(config.ROOT)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["hmdb51", "kth", "gmdcsa24"])
    p.add_argument("folder", type=Path)
    p.add_argument("--with-transitions", action="store_true", help="HMDB51: also import sit/stand")
    p.add_argument("--limit-per-source", type=int, default=0, help="max clips per movie/person (0 = all)")
    args = p.parse_args()
    if not args.folder.exists():
        sys.exit(f"Not found: {args.folder}")
    info = load_csv(INFO_CSV, "clip")
    if args.dataset == "hmdb51":
        import_hmdb51(args.folder, args.with_transitions, args.limit_per_source, info)
    elif args.dataset == "gmdcsa24":
        import_gmdcsa24(args.folder, info)
    else:
        import_kth(args.folder, args.limit_per_source)
    save_info(info)
    print("Next: python scripts/make_splits.py")


if __name__ == "__main__":
    main()
