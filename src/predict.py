"""Milestone 2 entry point, kept so old commands still work. Use src/run.py.

    python src/predict.py --input clip.mp4 --output outputs/
    python src/predict.py --input 0          # webcam
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.run import main as run_main  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--input" in argv:
        i = argv.index("--input") + 1
        if i < len(argv) and argv[i].isdigit():
            argv[i - 1:i + 1] = ["--webcam", argv[i]]
    run_main(argv)


if __name__ == "__main__":
    main()
