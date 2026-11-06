"""Convert the manuscript's Typst HTML export into an editable Word document.

Typst has no DOCX exporter, so the paper goes Typst -> HTML -> Pandoc -> DOCX.
The HTML export draws each figure as one inline composite ``<svg>``, which Pandoc
cannot place in Word. Each one is rasterised to a PNG with Typst itself (already
needed for the PDF, so no extra tools such as rsvg-convert) and swapped for an
``<img>`` before Pandoc runs.

    python scripts/build_docx.py manuscript.html manuscript.docx [--font-path DIR] [--ppi 200]
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SVG = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL)
SVG_URI = re.compile(r"data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)")
VIEWBOX = re.compile(r'viewBox="([-\d.e ]+)"')
PAD = 6  # pt of margin, as the figures draw panel letters outside their own box
STYLE = re.compile(r"\sstyle=\"[^\"]*\"")
TYPST_IMAGE = '#set page(width: auto, height: auto, margin: 0pt)\n#image("{svg}")\n'


def compile_png(svg: str, out: Path, font_path: str | None, ppi: int) -> None:
    """Render one standalone SVG document to ``out`` (PNG) using Typst."""
    svg_file = out.with_suffix(".svg")
    svg_file.write_text(svg, encoding="utf-8")
    doc = out.with_suffix(".typ")
    doc.write_text(TYPST_IMAGE.format(svg=svg_file.name), encoding="utf-8")
    cmd = ["typst", "compile", "--format", "png", "--ppi", str(ppi)]
    if font_path:
        cmd += ["--font-path", font_path]
    subprocess.run([*cmd, str(doc), str(out)], check=True, capture_output=True)


def rasterise(svg: str, out: Path, font_path: str | None, ppi: int) -> None:
    """Render a figure's composite SVG to ``out``.

    Typst draws text in SVGs nested inside an SVG with no font, so each embedded panel
    is first rendered to a PNG on its own and the figure is drawn from those.
    """

    def panel(match: re.Match[str]) -> str:
        data = base64.b64decode(match.group(1))
        png = out.with_name(f"{out.stem}-panel-{panel.n:02d}.png")
        panel.n += 1
        compile_png(data.decode("utf-8"), png, font_path, ppi * 2)
        return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()

    panel.n = 0
    svg = SVG_URI.sub(panel, svg)
    # The inline style (em sizes, overflow) is for browsers; the viewBox is what matters here.
    svg = STYLE.sub("", svg, count=1)

    def pad(match: re.Match[str]) -> str:
        x, y, w, h = (float(v) for v in match.group(1).split())
        box = f"{x - PAD} {y - PAD} {w + 2 * PAD} {h + 2 * PAD}"
        return f'viewBox="{box}"'

    svg = VIEWBOX.sub(pad, svg, count=1)
    # width/height (pt) must follow the viewBox, or Typst scales the padded box down.
    svg = re.sub(
        r'(<svg\b[^>]*?\swidth=")([\d.]+)pt',
        lambda m: f"{m[1]}{float(m[2]) + 2 * PAD}pt",
        svg,
        count=1,
    )
    svg = re.sub(
        r'(<svg\b[^>]*?\sheight=")([\d.]+)pt',
        lambda m: f"{m[1]}{float(m[2]) + 2 * PAD}pt",
        svg,
        count=1,
    )
    compile_png(svg, out, font_path, ppi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("html", type=Path, help="Typst HTML export of the manuscript")
    ap.add_argument("docx", type=Path, help="Word file to write")
    ap.add_argument("--font-path", help="extra font directory for Typst")
    ap.add_argument("--ppi", type=int, default=200, help="figure resolution")
    ap.add_argument("--reference-doc", help="Word file whose styles Pandoc should use")
    args = ap.parse_args()

    html = args.html.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        count = 0

        def swap(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            png = work / f"figure-{count:02d}.png"
            rasterise(match.group(0), png, args.font_path, args.ppi)
            return f'<img src="{png.name}" alt="Figure panel {count}">'

        html = SVG.sub(swap, html)
        page = work / "manuscript.html"
        page.write_text(html, encoding="utf-8")

        # `-smart` keeps the typographic quotes Typst already produced as they are.
        cmd = ["pandoc", str(page), "-f", "html-smart", "-o", str(args.docx.resolve())]
        cmd += ["--resource-path", str(work)]
        if args.reference_doc:
            cmd += ["--reference-doc", args.reference_doc]
        subprocess.run(cmd, check=True, cwd=work)

    print(f"wrote {args.docx} ({count} figures rasterised)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
