"""Split data/raw into data/{train,val,test}/<activity>/ BY SOURCE, not by file.

Clip names are <source>__<n>.<ext>. Every clip of a source goes to the same
split. Sources are shuffled with a fixed seed and assigned greedily so each split
gets roughly 70/15/15 of the clips. The assignment is saved to data/splits.csv.

Once data/splits.csv exists the test sources are LOCKED: re-running only places
NEW sources, it never moves an existing source between splits.

    python scripts/make_splits.py
"""
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

DATA = config.ROOT / "data"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SEED = 106


def source_of(path: Path) -> str:
    return path.stem.split("__")[0]


def main():
    clips = [p for p in (DATA / "raw").rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not clips:
        sys.exit("No clips in data/raw/<activity>/. Record some with scripts/record_clips.py")

    by_source = defaultdict(list)
    for c in clips:
        by_source[source_of(c)].append(c)

    splits_file = DATA / "splits.csv"
    assigned = {}
    if splits_file.exists():
        with open(splits_file) as f:
            assigned = {row["source"]: row["split"] for row in csv.DictReader(f)}

    counts = defaultdict(int)
    for src, split in assigned.items():
        counts[split] += len(by_source.get(src, []))
    new_sources = sorted(s for s in by_source if s not in assigned)
    random.Random(SEED).shuffle(new_sources)
    total = len(clips)
    for src in new_sources:
        # Put the source where the split is furthest below its target share.
        split = min(RATIOS, key=lambda s: counts[s] / total - RATIOS[s])
        assigned[src] = split
        counts[split] += len(by_source[src])

    for split in RATIOS:
        shutil.rmtree(DATA / split, ignore_errors=True)
    for src, files in by_source.items():
        for c in files:
            dest = DATA / assigned[src] / c.parent.name / c.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c, dest)

    with open(splits_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "clips"])
        for src in sorted(assigned):
            w.writerow([src, assigned[src], len(by_source.get(src, []))])

    print(f"{total} clips from {len(by_source)} sources")
    for split in RATIOS:
        per_class = defaultdict(int)
        for c in (DATA / split).rglob("*"):
            if c.suffix.lower() in VIDEO_EXTS:
                per_class[c.parent.name] += 1
        print(f"  {split:5s} {counts[split]:3d} clips  {dict(sorted(per_class.items()))}")


if __name__ == "__main__":
    main()
