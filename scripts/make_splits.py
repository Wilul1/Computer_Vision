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


def dataset_of(source: str) -> str:
    """'hmdb-batman' -> 'hmdb', 'gmdcsa-s1' -> 'gmdcsa'; our own sessions ('s01-...') -> 'own'."""
    prefix = source.split("-")[0]
    return prefix if prefix in ("hmdb", "gmdcsa", "kth") else "own"


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

    # Split each dataset on its own, so every split gets some of each kind of
    # footage (e.g. the 4 GMDCSA subjects must not all land in train).
    groups = defaultdict(list)
    for src in sorted(by_source):
        groups[dataset_of(src)].append(src)
    for group, sources in groups.items():
        g_counts = defaultdict(int)
        for s in sources:
            if s in assigned:
                g_counts[assigned[s]] += len(by_source[s])
        new_sources = [s for s in sources if s not in assigned]
        random.Random(f"{SEED}-{group}").shuffle(new_sources)
        g_total = sum(len(by_source[s]) for s in sources)
        for src in new_sources:
            # With 3+ sources, test and val each get at least one; after that,
            # the split furthest below its target share gets the next source.
            empty = [s for s in ("test", "val") if g_counts[s] == 0]
            if empty and len(sources) >= 3:
                split = empty[0]
            else:
                split = min(RATIOS, key=lambda s: g_counts[s] / g_total - RATIOS[s])
            assigned[src] = split
            g_counts[split] += len(by_source[src])
            counts[split] += len(by_source[src])

    for split in RATIOS:
        shutil.rmtree(DATA / split, ignore_errors=True)
        (DATA / split).mkdir()
        (DATA / split / ".gitkeep").touch()  # keeps the empty folder in git
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

    print(f"{len(clips)} clips from {len(by_source)} sources")
    for split in RATIOS:
        per_class = defaultdict(int)
        for c in (DATA / split).rglob("*"):
            if c.suffix.lower() in VIDEO_EXTS:
                per_class[c.parent.name] += 1
        print(f"  {split:5s} {counts[split]:3d} clips  {dict(sorted(per_class.items()))}")


if __name__ == "__main__":
    main()
