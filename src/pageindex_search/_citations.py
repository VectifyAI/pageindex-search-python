"""Citations as an agent with these tools writes them, as PageIndex's chat does: a tag after each sentence,
<cite doc="report.pdf" page="12"/>, and, asked for with CITATIONS_WITH_BOXES, the part of the page the answer is in,
box="x0,y0,x1,y1" (fractions of the page's width and height from its top left, as zoom_page takes them), to show where
it is. For that the agent looks at each page it cites: a few seconds more, and a picture's tokens.

    instructions = "Answer from the documents. " + CITATIONS_WITH_BOXES
    for c in citations(answer): c.document, c.page, c.box
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CITATIONS = ("After each sentence drawn from the documents, cite where, with a tag for each page: "
             '<cite doc="file name" page="12"/>, the file name as the tools give it.')
CITATIONS_WITH_BOXES = (CITATIONS + ' Add to each the part of the page the answer is in as box="x0,y0,x1,y1", fractions '
                        "of the page's width and height from its top left, as zoom_page takes them: before you cite a "
                        "page whose picture you have not seen, look at it with view_pages.")

# Taken as models vary the tag: in single quotes or none, in capitals, closed or not, a range or a list of pages, or
# in backticks.
_TAG = re.compile(r"`?<cite\b([^>]*?)/?>(?:\s*</cite>)?`?", re.IGNORECASE)


@dataclass
class Citation:
    """A page an answer cites: the document's file name, the page (from 1), the part of it if named, and where the tag
    is in the answer (start, end), to replace it with a link. document_id once found, by client.citations()."""
    document: str
    page: int
    box: tuple[float, float, float, float] | None = None
    start: int = 0
    end: int = 0
    document_id: str | None = None


def citations(text: str) -> list[Citation]:
    """The citations in an answer, in order: one for each page a tag names."""
    out: list[Citation] = []
    for m in _TAG.finditer(text):
        attributes = m.group(1)
        pages = _pages(_attribute(attributes, "page") or _attribute(attributes, "pages"))
        box = _box(_attribute(attributes, "box")) if len(pages) == 1 else None
        document = _attribute(attributes, "doc") or _attribute(attributes, "document")
        out.extend(Citation(document, page, box, m.start(), m.end()) for page in pages)
    return out


def _attribute(attributes: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s/>]+))", attributes, re.IGNORECASE)
    return next((g for g in m.groups() if g is not None), "").strip() if m else ""


def _pages(value: str) -> list[int]:
    """12, or 12 and 13 of "12-13", or 12 and 15 of "12, 15"."""
    pages: list[int] = []
    for part in re.split(r"\s*(?:,|;|and)\s*", value):
        span = re.fullmatch(r"(\d+)\s*[–—-]\s*(\d+)", part)
        if span and 0 <= int(span[2]) - int(span[1]) < 20:
            pages.extend(range(int(span[1]), int(span[2]) + 1))
        elif part.isdigit() and int(part) > 0:
            pages.append(int(part))
    return pages


def _box(value: str) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in re.split(r"[\s,]+", value.strip()) if v)
    except ValueError:  # not four numbers, or none
        return None
    return (x0, y0, x1, y1) if 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 else None
