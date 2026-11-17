"""Add a table of contents to the manuscript's HTML page.

Typst's HTML export has no outline, and gives an ``id`` only to the headings that are
referenced from elsewhere. This reads every ``<h2>``/``<h3>``, gives the ones without an
``id`` a slug, and inserts a ``<details>`` listing them: a sticky sidebar on wide screens,
a collapsible block on narrow ones. Existing ids are kept, so cross-references still work.

    python scripts/add_toc.py index.html
"""

from __future__ import annotations

import argparse
import re
import sys
from html import escape, unescape
from pathlib import Path

HEADING = re.compile(r"<h([23])((?:\s[^>]*)?)>(.*?)</h\1>", re.DOTALL)
ID = re.compile(r'\sid="([^"]*)"')
TAG = re.compile(r"<[^>]+>")
LINKS = re.compile(r'<p class="keywords">(?:(?!</p>).)*zephyr\.pdf.*?</p>', re.DOTALL)

BREAKPOINT = "1500px"
STYLE = f"""<style>
#toc{{font-size:.85rem;line-height:1.35;margin:1rem 0}}
#toc summary{{cursor:pointer;font-weight:600}}
#toc ol{{list-style:none;margin:.4rem 0 0;padding:0}}
#toc li{{margin:.15rem 0}}
#toc li.l3{{padding-left:1rem}}
#toc a{{text-decoration:none}}
#toc a:hover{{text-decoration:underline}}
@media (min-width:{BREAKPOINT}){{
#toc{{position:fixed;top:1.5rem;left:1.5rem;width:16rem;max-height:calc(100vh - 3rem);overflow:auto;margin:0}}
}}
</style>"""
OPEN_WHEN_WIDE = (
    f"<script>if(matchMedia('(min-width:{BREAKPOINT})').matches)"
    "document.getElementById('toc').open=true</script>"
)


def slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "section"
    out, n = base, 2
    while out in taken:
        out, n = f"{base}-{n}", n + 1
    taken.add(out)
    return out


def add_toc(html: str) -> str:
    taken = set(ID.findall(html))
    entries: list[tuple[int, str, str]] = []

    def tag(m: re.Match) -> str:
        level, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
        text = re.sub(r"\s+", " ", unescape(TAG.sub("", inner))).strip()
        found = ID.search(attrs)
        hid = found.group(1) if found else slug(text, taken)
        if not found:
            attrs += f' id="{hid}"'
        entries.append((level, hid, text))
        return f"<h{level}{attrs}>{inner}</h{level}>"

    html = HEADING.sub(tag, html)
    if not entries:
        sys.exit("add_toc: no <h2>/<h3> headings found")

    items = "".join(
        f'<li class="l{level}"><a href="#{hid}">{escape(text)}</a></li>'
        for level, hid, text in entries
    )
    toc = f'{STYLE}<details id="toc"><summary>Contents</summary><ol>{items}</ol></details>{OPEN_WHEN_WIDE}'

    # After the PDF/Word links when they are there, else straight after <body>.
    m = LINKS.search(html) or re.search(r"<body[^>]*>", html)
    if not m:
        sys.exit("add_toc: no <body> found")
    return html[: m.end()] + toc + html[m.end() :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("html", type=Path, help="the page to modify in place")
    path = parser.parse_args().html
    path.write_text(add_toc(path.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
