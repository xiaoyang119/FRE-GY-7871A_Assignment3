"""Render a Markdown report to PDF: Markdown -> styled HTML -> headless Chrome.

Needs `markdown-it-py` (for the Markdown) and Chrome or Edge (for the print).
No pandoc/LaTeX required.

    python scripts/make_report_pdf.py [REPORT.md] [report.pdf]
"""
import os
import pathlib
import shutil
import subprocess
import sys

CSS = """
@page { size: A4; margin: 15mm 14mm; }
html { -webkit-print-color-adjust: exact; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
       font-size: 10pt; line-height: 1.5; color: #111; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 5pt; border-bottom: 2px solid #333;
     padding-bottom: 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc;
     padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.5pt;
        table-layout: auto; }
th, td { border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left;
         vertical-align: top; word-wrap: break-word; }
th { background: #f0f0f0; font-weight: 600; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt;
       background: #f5f5f5; padding: 0 2pt; border-radius: 2px; }
pre { background: #f5f5f5; border: 0.5pt solid #ddd; padding: 6pt;
      font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3pt solid #bbb; margin: 6pt 0; padding: 2pt 10pt;
             color: #444; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 12pt 0; }
a { color: #0b5fa5; text-decoration: none; }
"""


def find_chrome():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for name in ("chrome", "msedge", "chromium"):
        w = shutil.which(name)
        if w:
            return w
    raise SystemExit("Chrome/Edge not found - install one or print the HTML manually")


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "REPORT.md")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "report.pdf")

    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    body = md.render(src.read_text(encoding="utf-8"))
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{src.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    html_path = out.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {html_path}")

    chrome = find_chrome()
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={out.resolve()}", html_path.resolve().as_uri()]
    print("running:", " ".join(cmd[:3]), "...")
    subprocess.run(cmd, check=True, timeout=180)
    if not out.exists():
        raise SystemExit("PDF was not produced")
    print(f"wrote {out.resolve()}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
