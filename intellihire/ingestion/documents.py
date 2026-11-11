"""Document ingestion engine: PyMuPDF (fitz) for PDFs and python-docx for Word.

Besides extracting text in a sensible reading order (including two-column
resumes), the extractor separates *hidden* text - white-on-white, microscopic
or off-page text - which is the usual carrier for resume prompt-injection and
keyword-stuffing attacks. Hidden text is reported, never scored.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field

from intellihire.ingestion.normalize import normalize_text

log = logging.getLogger(__name__)

try:  # "pymupdf" is the module name since PyMuPDF 1.24; "fitz" is the legacy alias
    import pymupdf as fitz

    HAS_PYMUPDF = True
except ImportError:  # pragma: no cover - depends on environment
    try:
        import fitz

        HAS_PYMUPDF = True
    except ImportError:
        fitz = None
        HAS_PYMUPDF = False


class DocumentParseError(Exception):
    """Raised when a document cannot be read."""


@dataclass
class ExtractedDocument:
    text: str
    hidden_text: str = ""
    pages: int = 1
    parser: str = ""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- PDF
_WHITE_THRESHOLD = 0xF0  # every RGB channel above this counts as "white"
_MIN_VISIBLE_FONT = 3.0  # points


def _is_white(color: int) -> bool:
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return min(r, g, b) >= _WHITE_THRESHOLD


_COLOR_MATCH_TOLERANCE = 0x18  # text within this per-channel distance of its background is invisible
_MAX_PDF_PAGES = 30  # resumes are short; a huge page count is abuse or the wrong file
PDF_PROCESSING_SECONDS = 20.0  # total wall-clock budget for extracting one PDF
MAX_DOCX_PARAGRAPHS = 5000
MAX_DOCX_CHARS = 200_000
_TOO_SLOW = "Document took too long to process"


@dataclass
class _Fill:
    """An opaque filled shape on a page."""

    x0: float
    y0: float
    x1: float
    y1: float
    rgb: tuple[int, int, int]
    dark: bool
    seqno: int  # paint-order position, shared numbering with get_texttrace(); -1 if unknown
    is_rect: bool  # True when the shape is made of rectangles/quads only

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def covers(self, bbox) -> bool:
        tol = 1.0
        return (
            self.x0 - tol <= bbox[0] and self.y0 - tol <= bbox[1]
            and self.x1 + tol >= bbox[2] and self.y1 + tol >= bbox[3]
        )


def _fill_rects(page) -> list[_Fill]:
    """Opaque filled shapes on a page in paint order.

    White text drawn on a dark sidebar or header band is legitimately visible -
    common in modern resume templates - and must not be quarantined as hidden.
    """
    rects: list[_Fill] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover - older PyMuPDF or malformed content
        return rects
    for d in drawings:
        fill, rect = d.get("fill"), d.get("rect")
        if not fill or rect is None or float(d.get("fill_opacity", 1.0) or 0.0) < 0.5:
            continue
        r, g, b = (list(fill) + [0.0, 0.0, 0.0])[:3]
        dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.6  # perceived luminance, 0-1
        items = d.get("items") or []
        is_rect = bool(items) and all(item and item[0] in ("re", "qu") for item in items)
        seqno = d.get("seqno")
        rects.append(
            _Fill(
                rect.x0, rect.y0, rect.x1, rect.y1,
                (round(r * 255), round(g * 255), round(b * 255)),
                dark,
                int(seqno) if seqno is not None else -1,
                is_rect,
            )
        )
    if rects and all(f.seqno >= 0 for f in rects):
        rects.sort(key=lambda f: f.seqno)
    return rects


def _text_seqnos(page) -> list[tuple[tuple[float, float, float, float], int]]:
    """(bbox, seqno) of every text-showing operation, to order spans against fills."""
    try:
        return [(tuple(t["bbox"]), int(t["seqno"])) for t in page.get_texttrace()]
    except Exception:  # pragma: no cover - older PyMuPDF
        return []


def _span_seqno_range(bbox, traces) -> tuple[int, int] | None:
    """First and last paint position of the text operations that drew this span."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    seqs = []
    for (tx0, ty0, tx1, ty1), seq in traces:
        tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
        if (tx0 <= cx <= tx1 and ty0 <= cy <= ty1) or (bbox[0] <= tcx <= bbox[2] and bbox[1] <= tcy <= bbox[3]):
            seqs.append(seq)
    return (min(seqs), max(seqs)) if seqs else None


def _background(bbox, fill_rects: list[_Fill], before: int | None = None) -> _Fill | None:
    """Topmost filled shape under the span's centre (painted before the text, when known)."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for f in reversed(fill_rects):
        if before is not None and 0 <= before <= f.seqno:
            continue
        if f.contains_point(cx, cy):
            return f
    return None


def _on_dark_background(bbox, fill_rects, before: int | None = None) -> bool:
    """True if the topmost filled shape under the span's centre is dark.

    Using the topmost shape (not "any dark shape") stops a white box painted
    over a dark one from being used to smuggle invisible white text through.
    """
    bg = _background(bbox, fill_rects, before)
    return bool(bg and bg.dark)


def _color_matches(color: int, rgb: tuple[int, int, int]) -> bool:
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return max(abs(r - rgb[0]), abs(g - rgb[1]), abs(b - rgb[2])) <= _COLOR_MATCH_TOLERANCE


def _covered_after(bbox, fill_rects: list[_Fill], after: int) -> bool:
    """True if an opaque rectangle painted *after* the text completely covers it."""
    return any(f.is_rect and f.seqno > after and f.covers(bbox) for f in fill_rects)


def _order_blocks(blocks: list[tuple[float, float, float, float, str]], page_width: float) -> list[str]:
    """Reading order for one- or two-column layouts.

    Blocks fully on the left half are read before blocks on the right half;
    full-width blocks (headers, summaries) act as separators between bands.
    """
    mid = page_width / 2
    tol = page_width * 0.04
    left = [b for b in blocks if b[2] <= mid + tol]
    right = [b for b in blocks if b[0] >= mid - tol and b[2] > mid + tol]
    two_columns = len(left) >= 3 and len(right) >= 3
    if not two_columns:
        return [b[4] for b in sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))]

    ordered: list[str] = []
    spanning = sorted((b for b in blocks if b not in left and b not in right), key=lambda b: b[1])
    cuts = [b[1] for b in spanning] + [float("inf")]
    start = float("-inf")
    for idx, cut in enumerate(cuts):
        band_left = sorted((b for b in left if start <= b[1] < cut), key=lambda b: b[1])
        band_right = sorted((b for b in right if start <= b[1] < cut), key=lambda b: b[1])
        ordered.extend(b[4] for b in band_left)
        ordered.extend(b[4] for b in band_right)
        if idx < len(spanning):
            ordered.append(spanning[idx][4])
        start = cut
    return ordered


def _extract_pdf_pymupdf(data: bytes) -> ExtractedDocument:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # fitz raises several exception types
        raise DocumentParseError(f"could not open PDF: {exc}") from exc
    if doc.needs_pass:
        raise DocumentParseError("PDF is password protected")

    if doc.page_count > _MAX_PDF_PAGES:
        page_total = doc.page_count
        doc.close()
        raise DocumentParseError(f"PDF has {page_total} pages; at most {_MAX_PDF_PAGES} pages are supported")

    visible_pages: list[str] = []
    hidden: list[str] = []
    deadline = time.monotonic() + PDF_PROCESSING_SECONDS
    with doc:
        for page in doc:
            if time.monotonic() > deadline:
                raise DocumentParseError(_TOO_SLOW)
            page_rect = page.rect
            fills: list[_Fill] | None = None  # computed lazily, once per page
            traces: list | None = None  # text paint order, only needed when the page has fills
            blocks: list[tuple[float, float, float, float, str]] = []
            for block in page.get_text("dict", sort=False).get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                lines_out: list[str] = []
                for line in block.get("lines", []):
                    visible_spans: list[str] = []
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        x0, y0, x1, y1 = bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                        off_page = x1 < page_rect.x0 or x0 > page_rect.x1 or y1 < page_rect.y0 or y0 > page_rect.y1
                        color = int(span.get("color", 0))
                        if fills is None:
                            fills = _fill_rects(page)
                        seq_range = None
                        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                        if any(f.contains_point(cx, cy) for f in fills):  # paint order only matters here
                            if traces is None:
                                traces = _text_seqnos(page)
                            seq_range = _span_seqno_range(bbox, traces)
                        bg = _background(bbox, fills, seq_range[0] if seq_range else None)
                        if _is_white(color):
                            # white text is only visible on a dark shape painted beneath it
                            camouflaged = not (bg and bg.dark)
                        else:
                            # text drawn in the colour of the rectangle beneath it (black on black)
                            camouflaged = bool(bg and bg.is_rect and _color_matches(color, bg.rgb))
                        covered = bool(seq_range) and _covered_after(bbox, fills, seq_range[1])
                        invisible = (
                            camouflaged
                            or covered
                            or float(span.get("size", 12)) < _MIN_VISIBLE_FONT
                            or int(span.get("alpha", 255)) < 30
                            or off_page
                        )
                        if invisible:
                            hidden.append(text)
                        else:
                            visible_spans.append(text)
                    if visible_spans:
                        lines_out.append("".join(visible_spans))
                if lines_out:
                    x0, y0, x1, y1 = block["bbox"]
                    blocks.append((x0, y0, x1, y1, "\n".join(lines_out)))
            visible_pages.append("\n".join(_order_blocks(blocks, page_rect.width)))
        page_count = doc.page_count

    text = normalize_text("\n\n".join(visible_pages))
    result = ExtractedDocument(
        text=text,
        hidden_text=normalize_text(" ".join(hidden)),
        pages=page_count,
        parser="pymupdf",
    )
    if len(text) < 50:
        result.warnings.append("very little text extracted - the PDF may be a scanned image")
    return result


def _extract_pdf_pypdf(data: bytes) -> ExtractedDocument:  # pragma: no cover - fallback
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentParseError("PyMuPDF is not installed (pip install pymupdf)") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > _MAX_PDF_PAGES:
            raise DocumentParseError(
                f"PDF has {len(reader.pages)} pages; at most {_MAX_PDF_PAGES} pages are supported"
            )
        deadline = time.monotonic() + PDF_PROCESSING_SECONDS
        page_texts: list[str] = []
        for page in reader.pages:
            if time.monotonic() > deadline:
                raise DocumentParseError(_TOO_SLOW)
            page_texts.append(page.extract_text() or "")
        text = "\n\n".join(page_texts)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"could not read PDF: {exc}") from exc
    return ExtractedDocument(
        text=normalize_text(text),
        pages=len(reader.pages),
        parser="pypdf (fallback - hidden-text detection unavailable)",
        warnings=["PyMuPDF not installed: hidden-text detection disabled"],
    )


def extract_pdf(data: bytes) -> ExtractedDocument:
    if HAS_PYMUPDF:
        return _extract_pdf_pymupdf(data)
    return _extract_pdf_pypdf(data)


# -------------------------------------------------------------------------- DOCX
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _w(tag: str) -> str:
    return f"{{{_W_NS}}}{tag}"


# Theme colours that resolve to the page background (white) under the default colour mapping.
_WHITE_THEME_COLORS = {"background1", "light1", "bg1", "lt1"}
_DARK_THEME_COLORS = {"text1", "dark1", "tx1", "dk1", "text2", "dark2", "tx2", "dk2"}
_DARK_HIGHLIGHTS = {
    "black", "blue", "darkBlue", "darkCyan", "darkGreen", "darkMagenta",
    "darkRed", "darkYellow", "darkGray", "red", "green", "magenta",
}
# Containers whose text is not shown: tracked deletions and the legacy duplicate of text boxes.
_SKIP_CONTAINERS = {_w("del"), _w("moveFrom"), f"{{{_MC_NS}}}Fallback"}
_RUN_CONTAINERS = {
    _w("hyperlink"), _w("ins"), _w("moveTo"), _w("smartTag"), _w("sdt"), _w("sdtContent"),
    _w("customXml"), _w("fldSimple"), _w("bdo"), _w("dir"),
}


def _on_off(el) -> bool | None:
    """Value of an OOXML on/off element (<w:vanish/>, <w:vanish w:val="0"/>), None if absent."""
    if el is None:
        return None
    return el.get(_w("val"), "true").lower() not in {"0", "false", "off"}


def _hex_rgb(value: str | None) -> tuple[int, int, int] | None:
    if not value or len(value) != 6:
        return None
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except ValueError:
        return None


def _is_white_color(color_el) -> bool:
    """True if a <w:color> element renders as white (direct RGB or a background theme colour)."""
    theme = color_el.get(_w("themeColor"))
    if theme:
        if theme not in _WHITE_THEME_COLORS:
            return False
        shade = color_el.get(_w("themeShade"))  # darkens the theme colour by shade/255
        if shade:
            try:
                return int(shade, 16) >= _WHITE_THRESHOLD
            except ValueError:
                return True
        return True
    rgb = _hex_rgb(color_el.get(_w("val")))
    return rgb is not None and min(rgb) >= _WHITE_THRESHOLD


def _is_dark_shading(shd) -> bool:
    if shd is None:
        return False
    theme = shd.get(_w("themeFill"))
    if theme:
        return theme in _DARK_THEME_COLORS
    rgb = _hex_rgb(shd.get(_w("fill")))
    if rgb is None:
        return False
    r, g, b = (c / 255 for c in rgb)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.6


class _DocxStyles:
    """Resolves effective run properties: direct rPr -> character style (rStyle, basedOn
    chain) -> paragraph style (pStyle or the default paragraph style, basedOn chain) ->
    docDefaults. The first level that defines a property wins."""

    def __init__(self, document) -> None:
        self.by_id: dict[str, object] = {}
        self.default_para: str | None = None
        self.defaults_rpr = None
        try:
            root = document.styles.element
        except Exception:  # pragma: no cover - no styles part
            return
        for st in root.iterchildren(_w("style")):
            sid = st.get(_w("styleId"))
            if sid:
                self.by_id[sid] = st
                if st.get(_w("type")) == "paragraph" and st.get(_w("default")) in {"1", "true", "on"}:
                    self.default_para = sid
        dd = root.find(_w("docDefaults"))
        if dd is not None:
            self.defaults_rpr = dd.find(f"{_w('rPrDefault')}/{_w('rPr')}")

    def _chain(self, style_id: str | None) -> list:
        out, seen = [], set()
        while style_id and style_id not in seen and style_id in self.by_id:
            seen.add(style_id)
            st = self.by_id[style_id]
            rpr = st.find(_w("rPr"))
            if rpr is not None:
                out.append(rpr)
            based = st.find(_w("basedOn"))
            style_id = based.get(_w("val")) if based is not None else None
        return out

    def rpr_chain(self, r, p) -> list:
        """rPr elements from highest to lowest precedence for run ``r`` in paragraph ``p``."""
        chain = []
        rpr = r.find(_w("rPr"))
        if rpr is not None:
            chain.append(rpr)
            rstyle = rpr.find(_w("rStyle"))
            if rstyle is not None:
                chain.extend(self._chain(rstyle.get(_w("val"))))
        pstyle_id = self.default_para
        ppr = p.find(_w("pPr")) if p is not None else None
        if ppr is not None:
            pstyle = ppr.find(_w("pStyle"))
            if pstyle is not None and pstyle.get(_w("val")) in self.by_id:
                pstyle_id = pstyle.get(_w("val"))
        chain.extend(self._chain(pstyle_id))
        if self.defaults_rpr is not None:
            chain.append(self.defaults_rpr)
        return chain


def _effective(chain: list, tag: str):
    for rpr in chain:
        el = rpr.find(_w(tag))
        if el is not None:
            return el
    return None


def _run_on_dark_background(r, p, chain) -> bool:
    """White text is legitimately visible on dark run/paragraph/table-cell shading."""
    hl = _effective(chain, "highlight")
    if hl is not None and hl.get(_w("val")) in _DARK_HIGHLIGHTS:
        return True
    if _is_dark_shading(_effective(chain, "shd")):
        return True
    if p is not None:
        ppr = p.find(_w("pPr"))
        if ppr is not None and _is_dark_shading(ppr.find(_w("shd"))):
            return True
        anc = p.getparent()
        while anc is not None:
            if anc.tag == _w("tc"):
                tcpr = anc.find(_w("tcPr"))
                if tcpr is not None and _is_dark_shading(tcpr.find(_w("shd"))):
                    return True
                break
            anc = anc.getparent()
    return False


def _docx_run_is_hidden(r, p, styles: _DocxStyles) -> bool:
    chain = styles.rpr_chain(r, p)
    if _on_off(_effective(chain, "vanish")) or _on_off(_effective(chain, "specVanish")):
        return True
    sz = _effective(chain, "sz")
    if sz is not None:
        try:
            if int(sz.get(_w("val"), "0")) < _MIN_VISIBLE_FONT * 2:  # half-points
                return True
        except ValueError:
            pass
    color = _effective(chain, "color")
    if color is not None and _is_white_color(color) and not _run_on_dark_background(r, p, chain):
        return True
    return False


def _paragraph_runs(p):
    """Runs shown in paragraph ``p`` in order: direct runs and runs inside hyperlinks, content
    controls, insertions etc. Nested paragraphs (text boxes) and deleted text are skipped."""
    stack = list(reversed(list(p)))
    while stack:
        el = stack.pop()
        tag = el.tag
        if tag == _w("r"):
            yield el
        elif tag in _RUN_CONTAINERS:
            stack.extend(reversed(list(el)))


def _iter_paragraphs(root):
    """Every w:p under ``root`` in document order - body, tables (nested too) and text boxes
    (w:txbxContent) - excluding the mc:Fallback duplicate of text boxes and deleted content."""
    stack = [root]
    while stack:
        el = stack.pop()
        if el.tag in _SKIP_CONTAINERS or not isinstance(el.tag, str):
            continue
        if el.tag == _w("p"):
            yield el
        stack.extend(reversed(list(el)))


def extract_docx(data: bytes) -> ExtractedDocument:
    try:
        import docx
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        from docx.text.run import Run
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("python-docx is not installed") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"could not open DOCX: {exc}") from exc

    visible: list[str] = []
    hidden: list[str] = []
    styles = _DocxStyles(document)

    counts = {"paragraphs": 0, "chars": 0}

    def handle_root(root) -> None:
        for p in _iter_paragraphs(root):
            counts["paragraphs"] += 1
            if counts["paragraphs"] > MAX_DOCX_PARAGRAPHS:
                raise DocumentParseError(
                    f"DOCX has more than {MAX_DOCX_PARAGRAPHS} paragraphs; this is not a normal resume"
                )
            shown: list[str] = []
            for r in _paragraph_runs(p):
                text = Run(r, None).text
                if not text:
                    continue
                counts["chars"] += len(text)
                if counts["chars"] > MAX_DOCX_CHARS:
                    raise DocumentParseError(
                        f"DOCX contains more than {MAX_DOCX_CHARS:,} characters of text; this is not a normal resume"
                    )
                (hidden if _docx_run_is_hidden(r, p, styles) else shown).append(text)
            line = "".join(shown).strip()
            if line:
                visible.append(line)

    handle_root(document.element.body)
    # Headers then footers, each part once (sections may share them).
    for reltype in (RT.HEADER, RT.FOOTER):
        for rel in list(document.part.rels.values()):
            if rel.is_external or rel.reltype != reltype:
                continue
            try:
                handle_root(rel.target_part.element)
            except DocumentParseError:
                raise
            except Exception:  # pragma: no cover - malformed header/footer part
                log.debug("skipping unreadable DOCX header/footer part", exc_info=True)

    text = normalize_text("\n".join(visible))
    return ExtractedDocument(
        text=text,
        hidden_text=normalize_text(" ".join(hidden)),
        pages=1,
        parser="python-docx",
    )


# --------------------------------------------------------------------------- TXT
def extract_txt(data: bytes) -> ExtractedDocument:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return ExtractedDocument(text=normalize_text(data.decode(encoding)), parser="text")
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("unreadable text encoding")  # pragma: no cover


def extract_document(filename: str, data: bytes) -> ExtractedDocument:
    """Dispatch on extension. The file must already have passed validation."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        doc = extract_pdf(data)
    elif lower.endswith(".docx"):
        doc = extract_docx(data)
    elif lower.endswith(".txt"):
        doc = extract_txt(data)
    else:
        raise DocumentParseError(f"unsupported document type: {filename}")
    if not doc.text.strip():
        raise DocumentParseError("no readable text found in document")
    return doc
