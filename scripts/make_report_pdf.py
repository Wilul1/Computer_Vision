"""Turn docs/progress_report.md into a PDF for the hand-in.

    python scripts/make_report_pdf.py
    python scripts/make_report_pdf.py --input docs/design_choices.md

Converts Markdown to HTML, then prints it with Microsoft Edge or Google Chrome in
headless mode (both are on most Windows laptops). Output goes next to the input
as <name>.pdf.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chrome",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f4f4f4; padding: 0 2pt; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3px solid #999; color: #444; background: #fafafa; }
img { max-width: 100%; }
"""


def find_browser():
    for b in BROWSERS:
        path = shutil.which(b) or (b if Path(b).exists() else None)
        if path:
            return path
    sys.exit("No Edge or Chrome found. Open the .md in VS Code and use 'Markdown PDF', or print the HTML yourself.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "docs" / "progress_report.md"))
    p.add_argument("--output", default=None)
    args = p.parse_args()
    src = Path(args.input).resolve()
    out = Path(args.output).resolve() if args.output else src.with_suffix(".pdf")

    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{src.stem}</title>" \
           f"<base href='{src.parent.as_uri()}/'><style>{CSS}</style></head><body>{body}</body></html>"

    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp}\\profile", f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    if not out.exists():
        sys.exit("The browser did not write the PDF.")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
