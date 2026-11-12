"""Rebuild data/ on a new machine in one command (≈3.2 GB of downloads).

    python scripts/get_data.py                       # downloads to ../datasets next to the repo
    python scripts/get_data.py --root D:/datasets    # or anywhere else

Steps: download HMDB51 (Hugging Face mirror) and extract only walk/ and run/;
download GMDCSA24 from GitHub; run import_public.py for both; run make_splits.py.
data/splits.csv is committed, so the split comes out identical.

You do NOT need this to reproduce the reported numbers: evaluate.py scores the
committed per-frame features in outputs/cache/ when the videos are missing.
Use it to re-run the pipeline on the real videos or to make the demo inputs.
"""
import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HMDB_URL = "https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip"
GMDCSA_URL = "https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/archive/refs/heads/master.zip"


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n  -> {dest}")
    tmp = dest.with_suffix(".part")

    def progress(blocks, block_size, total):
        if total > 0 and blocks % 200 == 0:
            print(f"    {blocks * block_size / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
    urllib.request.urlretrieve(url, tmp, progress)
    tmp.rename(dest)


def run(*args):
    print("  $ python " + " ".join(map(str, args)))
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT.parent / "datasets", help="where to keep the downloads")
    args = ap.parse_args()
    root = args.root.resolve()

    print("1/4 HMDB51 (walk, run)")
    hmdb_zip = root / "hmdb51" / "hmdb51.zip"
    download(HMDB_URL, hmdb_zip)
    with zipfile.ZipFile(hmdb_zip) as z:
        members = [m for m in z.namelist()
                   if m.endswith(".avi") and m.split("/")[1:2] and m.split("/")[1] in ("walk", "run")]
        z.extractall(root / "hmdb51", members=members)
    print(f"  extracted {len(members)} clips")

    print("2/4 GMDCSA24")
    gm_zip = root / "GMDCSA24.zip"
    gm_dir = root / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master"
    if not gm_dir.exists():
        download(GMDCSA_URL, gm_zip)
        with zipfile.ZipFile(gm_zip) as z:
            z.extractall(root)

    print("3/4 import into data/raw")
    run("scripts/import_public.py", "hmdb51", root / "hmdb51" / "hmdb51", "--limit-per-source", "3")
    run("scripts/import_public.py", "gmdcsa24", gm_dir)

    print("4/4 split (uses the committed data/splits.csv)")
    run("scripts/make_splits.py")
    print("Done. Next: python src/evaluate.py --split test --rerun   (re-runs the pipeline on the videos)")


if __name__ == "__main__":
    main()
