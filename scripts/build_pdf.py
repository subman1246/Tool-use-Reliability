"""Build a submittable PDF from the markdown draft.

No pandoc or LaTeX is available in this environment, so the path is
markdown -> self-contained HTML -> headless-Chrome print-to-PDF. This is the fastest
route that preserves document structure: numbered sections, tables, figures and the
reference list all survive, and MathJax renders the inline LaTeX.

This is deliberately NOT the venue's template. The job here is a complete, correct,
readable document; swapping in a specific class file is mechanical and belongs after
the deadline rather than in front of it.
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import re
import subprocess
import sys

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Georgia","Times New Roman",serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt 0; }
h2 { font-size: 13pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #bbb;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt 0; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt 0; page-break-after: avoid; }
p { margin: 0 0 7pt 0; text-align: justify; }
code { font-family: "Consolas","Courier New",monospace; font-size: 9pt;
       background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; border-left: 3px solid #ccc; padding: 6pt 8pt;
      font-size: 8.5pt; overflow-x: auto; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt 0;
        font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #eee; font-weight: bold; }
blockquote { margin: 8pt 0 8pt 12pt; padding-left: 10pt;
             border-left: 3px solid #888; color: #222; font-style: normal; }
img { max-width: 100%; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ddd; margin: 14pt 0; }
ul, ol { margin: 0 0 7pt 0; padding-left: 20pt; }
li { margin-bottom: 2pt; }
"""

MATHJAX = """
<script>
window.MathJax = { tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
                   options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] } };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def embed_images(html: str, root: pathlib.Path) -> str:
    """Inline figures as data URIs so the PDF never depends on file paths."""
    def repl(m):
        src = m.group(1)
        if src.startswith(("http", "data:")):
            return m.group(0)
        for cand in (root / src, root / "data" / "results" / "figures" / pathlib.Path(src).name):
            if cand.exists():
                b64 = base64.b64encode(cand.read_bytes()).decode()
                return m.group(0).replace(src, f"data:image/png;base64,{b64}")
        return m.group(0)
    return re.sub(r'<img[^>]+src="([^"]+)"', repl, html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/PAPER_DRAFT.md")
    ap.add_argument("--out", default="docs/paper.pdf")
    ap.add_argument("--title", default="Invocation-Level Reliability of Tool Use in "
                                       "Language Model Agents")
    args = ap.parse_args()

    root = pathlib.Path(".").resolve()
    src = pathlib.Path(args.src)
    text = src.read_text(encoding="utf-8")

    import markdown
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"])
    body = embed_images(body, root)

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{args.title}</title><style>{CSS}</style>{MATHJAX}</head>"
            f"<body>{body}</body></html>")
    html_path = pathlib.Path(args.out).with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {html_path} ({html_path.stat().st_size:,} bytes)")

    browser = next((c for c in CHROME_CANDIDATES if pathlib.Path(c).exists()), None)
    if browser is None:
        print("no headless browser found; HTML written, PDF not built", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out).resolve()
    import tempfile
    profile = tempfile.mkdtemp(prefix="chrome-pdf-")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
           f"--user-data-dir={profile}",
           "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=30000",
           f"--print-to-pdf={out}", "--no-pdf-header-footer",
           html_path.resolve().as_uri()]
    print("running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if out.exists():
        print(f"PDF: {out}  ({out.stat().st_size:,} bytes)")
        return 0
    print("PDF not produced", r.returncode, r.stderr[-800:], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
