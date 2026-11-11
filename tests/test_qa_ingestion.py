"""Regression tests for QA findings owned by the ingestion team (T03, T14, T18, T34, T36, T37, T41, T42)."""
from __future__ import annotations

import io

import pytest

from intellihire.ingestion.documents import DocumentParseError, extract_document, extract_docx, extract_pdf
from intellihire.ingestion.normalize import normalize_text
from intellihire.ingestion.validation import file_stem_key, sanitize_filename, validate_document

STUFF = "Kubernetes Terraform AWS"
VISIBLE = "Jane Smith visible resume line with enough text to pass the check"


# ----------------------------------------------------------------------------- DOCX helpers
def _docx_bytes(doc) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _new_docx():
    docx = pytest.importorskip("docx")
    return docx.Document()


# ----------------------------------------------------------------------------- T03
@pytest.mark.parametrize("variant", ["vanish", "white", "tiny"])
def test_t03_docx_character_style_hides_text(variant):
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Pt, RGBColor

    doc = _new_docx()
    st = doc.styles.add_style("Stealth", WD_STYLE_TYPE.CHARACTER)
    if variant == "vanish":
        st.font.hidden = True
    elif variant == "white":
        st.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    else:
        st.font.size = Pt(1)
    p = doc.add_paragraph("Jane Smith visible line. ")
    p.add_run(STUFF, style="Stealth")
    result = extract_docx(_docx_bytes(doc))
    assert "Kubernetes" not in result.text
    assert STUFF in result.hidden_text


def test_t03_docx_character_style_based_on_chain():
    from docx.enum.style import WD_STYLE_TYPE

    doc = _new_docx()
    base = doc.styles.add_style("BaseHidden", WD_STYLE_TYPE.CHARACTER)
    base.font.hidden = True
    child = doc.styles.add_style("ChildStyle", WD_STYLE_TYPE.CHARACTER)
    child.base_style = base
    doc.add_paragraph("Jane Smith visible line. ").add_run(STUFF, style="ChildStyle")
    result = extract_docx(_docx_bytes(doc))
    assert "Kubernetes" not in result.text and STUFF in result.hidden_text


def test_t03_docx_paragraph_style_white_and_spec_vanish():
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import OxmlElement
    from docx.shared import RGBColor

    doc = _new_docx()
    st = doc.styles.add_style("WhitePara", WD_STYLE_TYPE.PARAGRAPH)
    st.font.color.rgb = RGBColor(255, 255, 255)
    doc.add_paragraph("Jane Smith visible line.")
    doc.add_paragraph(STUFF, style="WhitePara")
    run = doc.add_paragraph("Also visible. ").add_run("Kafka Airflow")
    run._r.get_or_add_rPr().append(OxmlElement("w:specVanish"))
    result = extract_docx(_docx_bytes(doc))
    assert "Kubernetes" not in result.text and STUFF in result.hidden_text
    assert "Kafka" not in result.text and "Kafka" in result.hidden_text
    assert "Also visible." in result.text


def test_t03_docx_doc_defaults_vanish():
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = _new_docx()
    doc.add_paragraph(STUFF)
    styles = doc.styles.element
    dd = styles.find(qn("w:docDefaults"))
    rpr = dd.find(qn("w:rPrDefault")).find(qn("w:rPr"))
    rpr.append(OxmlElement("w:vanish"))
    # Normal style must not override it
    result = extract_docx(_docx_bytes(doc))
    assert STUFF in result.hidden_text and "Kubernetes" not in result.text


@pytest.mark.parametrize("theme", ["background1", "light1", "bg1"])
def test_t03_docx_theme_background_colour_is_hidden(theme):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = _new_docx()
    run = doc.add_paragraph("Jane Smith visible line. ").add_run(STUFF)
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "000000")
    c.set(qn("w:themeColor"), theme)
    run._r.get_or_add_rPr().append(c)
    result = extract_docx(_docx_bytes(doc))
    assert "Kubernetes" not in result.text and STUFF in result.hidden_text


def test_t03_docx_no_false_positives_for_normal_styling():
    """Shaded-theme grey, white text in a dark table cell and explicit un-hiding stay visible."""
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import RGBColor

    doc = _new_docx()
    run = doc.add_paragraph().add_run("Grey theme text Python")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "BFBFBF")
    c.set(qn("w:themeColor"), "background1")
    c.set(qn("w:themeShade"), "BF")
    run._r.get_or_add_rPr().append(c)

    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "1F2937")
    tcpr.append(shd)
    cell.paragraphs[0].add_run("Header band Docker").font.color.rgb = RGBColor(255, 255, 255)

    hid = doc.styles.add_style("HiddenBase", WD_STYLE_TYPE.CHARACTER)
    hid.font.hidden = True
    r = doc.add_paragraph().add_run("Unhidden Django", style="HiddenBase")
    r.font.hidden = False  # direct formatting overrides the style
    result = extract_docx(_docx_bytes(doc))
    assert result.hidden_text == ""
    for word in ("Python", "Docker", "Django"):
        assert word in result.text


# ----------------------------------------------------------------------------- T42
def test_t42_docx_footer_textbox_and_nested_table():
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    doc = _new_docx()
    doc.add_paragraph("Jane Smith visible line.")
    doc.sections[0].footer.paragraphs[0].text = "Footer: Python Django"

    outer = doc.add_table(rows=1, cols=1)
    inner = outer.cell(0, 0).add_table(rows=1, cols=1)
    inner.cell(0, 0).paragraphs[0].text = "Nested cell: Spark Airflow"

    # A text box written the way Word does it: drawingML choice + VML fallback duplicate.
    txbx = parse_xml(
        f'<w:r {nsdecls("w")} xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
        ' xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
        ' xmlns:v="urn:schemas-microsoft-com:vml">'
        "<mc:AlternateContent><mc:Choice Requires=\"wps\"><w:drawing><wps:wsp><wps:txbx><w:txbxContent>"
        "<w:p><w:r><w:t>Textbox: GraphQL Redis</w:t></w:r></w:p>"
        "</w:txbxContent></wps:txbx></wps:wsp></w:drawing></mc:Choice>"
        "<mc:Fallback><w:pict><v:textbox><w:txbxContent>"
        "<w:p><w:r><w:t>Textbox: GraphQL Redis</w:t></w:r></w:p>"
        "</w:txbxContent></v:textbox></w:pict></mc:Fallback></mc:AlternateContent></w:r>"
    )
    doc.add_paragraph()._p.append(txbx)

    result = extract_docx(_docx_bytes(doc))
    assert "Footer: Python Django" in result.text
    assert "Nested cell: Spark Airflow" in result.text
    assert result.text.count("Textbox: GraphQL Redis") == 1  # VML fallback not duplicated
    assert result.hidden_text == ""


def test_t42_docx_hidden_text_in_footer_is_quarantined():
    doc = _new_docx()
    doc.add_paragraph("Jane Smith visible line.")
    run = doc.sections[0].footer.paragraphs[0].add_run(STUFF)
    run.font.hidden = True
    result = extract_docx(_docx_bytes(doc))
    assert "Kubernetes" not in result.text and STUFF in result.hidden_text


# ----------------------------------------------------------------------------- T14
def _pdf(build) -> bytes:
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    page = doc.new_page()
    build(fitz, page)
    return doc.tobytes()


def test_t14_pdf_black_text_on_black_fill_is_hidden():
    def build(fitz, page):
        page.insert_text((72, 72), VISIBLE, fontsize=11)
        page.draw_rect(fitz.Rect(60, 280, 400, 310), color=(0, 0, 0), fill=(0, 0, 0))
        page.insert_text((72, 300), STUFF, fontsize=11, color=(0, 0, 0))

    doc = extract_pdf(_pdf(build))
    assert STUFF in doc.hidden_text and "Kubernetes" not in doc.text
    assert "Jane Smith" in doc.text


def test_t14_pdf_text_covered_by_later_opaque_rect_is_hidden():
    def build(fitz, page):
        page.insert_text((72, 72), VISIBLE, fontsize=11)
        page.insert_text((72, 300), STUFF, fontsize=11, color=(0, 0, 0))
        page.draw_rect(fitz.Rect(60, 280, 400, 310), color=(1, 1, 1), fill=(1, 1, 1))

    doc = extract_pdf(_pdf(build))
    assert STUFF in doc.hidden_text and "Kubernetes" not in doc.text


def test_t14_pdf_normal_resume_shapes_are_not_false_positives():
    """Dark header band with white text, a coloured band with dark text, a highlight
    behind text, a semi-transparent overlay and an underline drawn after text all stay visible."""
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setFillColorRGB(0.1, 0.2, 0.4)
    c.rect(0, 760, 600, 82, fill=1, stroke=0)  # dark header bar
    c.setFillColorRGB(1, 1, 1)
    c.drawString(40, 800, "Jane Doe Senior Engineer Docker")  # white on dark -> visible
    c.setFillColorRGB(0.85, 0.9, 1.0)
    c.rect(0, 700, 600, 40, fill=1, stroke=0)  # light blue band
    c.setFillColorRGB(0.1, 0.1, 0.3)
    c.drawString(40, 715, "Skills Python Terraform")  # dark on light -> visible
    c.setFillColorRGB(1, 1, 0)
    c.rect(38, 645, 200, 16, fill=1, stroke=0)  # highlight behind
    c.setFillColorRGB(0, 0, 0)
    c.drawString(40, 650, "Highlighted Kafka")
    c.drawString(40, 600, "Underlined Airflow line")
    c.rect(40, 598, 150, 0.8, fill=1, stroke=0)  # underline drawn after the text
    c.drawString(40, 550, "Watermarked Spark text")
    c.setFillColorRGB(0.5, 0.5, 0.5, alpha=0.2)
    c.rect(30, 540, 300, 30, fill=1, stroke=0)  # translucent overlay after the text
    c.save()

    doc = extract_document("cv.pdf", buf.getvalue())
    assert doc.hidden_text == ""
    for word in ("Docker", "Terraform", "Kafka", "Airflow", "Spark"):
        assert word in doc.text


def test_t14_white_text_on_dark_bar_drawn_with_pymupdf_is_visible():
    def build(fitz, page):
        page.insert_text((72, 200), VISIBLE, fontsize=11)
        page.draw_rect(fitz.Rect(0, 40, 612, 100), color=None, fill=(0.15, 0.15, 0.2))
        page.insert_text((72, 75), "Header Kubernetes", fontsize=14, color=(1, 1, 1))

    doc = extract_pdf(_pdf(build))
    assert "Header Kubernetes" in doc.text and doc.hidden_text == ""


# ----------------------------------------------------------------------------- T41
def test_t41_pdf_page_cap():
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    for _ in range(31):
        doc.new_page().insert_text((72, 72), VISIBLE)
    with pytest.raises(DocumentParseError, match="pages"):
        extract_document("cv.pdf", doc.tobytes())

    ok = fitz.open()
    for _ in range(30):
        ok.new_page().insert_text((72, 72), VISIBLE)
    assert extract_document("cv.pdf", ok.tobytes()).pages == 30


# ----------------------------------------------------------------------------- T34
def test_t34_long_filename_keeps_extension():
    name = "a" * 150 + "_resume.pdf"
    safe = sanitize_filename(name)
    assert len(safe) <= 150 and safe.endswith(".pdf")
    assert validate_document(name, b"%PDF-1.7\n...").ok
    assert sanitize_filename("x" * 300) == "x" * 150


# ----------------------------------------------------------------------------- T36
def test_t36_hyphen_rejoin_rules():
    assert normalize_text("Worked 2019-\n2021") == "Worked 2019-\n2021"
    assert normalize_text("develop-\n  ment") == "development"
    assert normalize_text("Python-\nDjango") == "Python-\nDjango"
    assert normalize_text("develop-\nment") == "development"


# ----------------------------------------------------------------------------- T37 / T18
def test_t37_stem_key_keeps_cv_inside_words():
    assert file_stem_key("Cvijanovic.pdf") == "cvijanovic"
    assert file_stem_key("Cvijanovic_Marko_CV.pdf") == "cvijanovicmarko"
    assert file_stem_key("introvert_resume.pdf") == "introvert"
    assert file_stem_key("Jane_Doe_Resume.pdf") == file_stem_key("jane-doe-intro.mp4") == "janedoe"


def test_t18_stem_key_never_empty():
    for name in ("resume.pdf", "video.mp4", "CV.docx", "intro.mov", "___.pdf", ".pdf", ""):
        assert file_stem_key(name) != ""
    assert file_stem_key("resume.pdf") != file_stem_key("video.mp4")
