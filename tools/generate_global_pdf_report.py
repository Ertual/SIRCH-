from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "rapport_sirch_global.md"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "rapport_sirch_global_complet.pdf"


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "SirchTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0b1f33"),
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "SirchSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4d5f73"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "SirchH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=22,
            textColor=colors.HexColor("#0b1f33"),
            spaceBefore=18,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "SirchH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=18,
            textColor=colors.HexColor("#174a7c"),
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "SirchH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#1d6258"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "h4": ParagraphStyle(
            "SirchH4",
            parent=base["Heading4"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#333333"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "SirchBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.4,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "SirchBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.4,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "SirchCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.8,
            leading=9.5,
            leftIndent=6,
            rightIndent=6,
            backColor=colors.HexColor("#f3f6f8"),
            borderColor=colors.HexColor("#d6dde4"),
            borderWidth=0.5,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "SirchCaption",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#384858"),
            spaceBefore=4,
            spaceAfter=4,
        ),
    }
    return styles


def esc(text: str) -> str:
    text = text.replace("<br>", "\n")
    return html.escape(text).replace("`", "")


def inline_markdown(text: str) -> str:
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def parse_table(rows: list[str], doc_width: float, styles) -> Table:
    parsed: list[list[str]] = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
            continue
        parsed.append(cells)

    if not parsed:
        return Table([[""]])

    col_count = max(len(r) for r in parsed)
    for row in parsed:
        row.extend([""] * (col_count - len(row)))

    font_size = 7.0
    leading = 8.3
    if col_count > 6:
        font_size = 5.7
        leading = 6.8
    if col_count > 9:
        font_size = 4.8
        leading = 5.9

    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        "TableHead",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    data = []
    for i, row in enumerate(parsed):
        style = head_style if i == 0 else cell_style
        data.append([Paragraph(inline_markdown(cell), style) for cell in row])

    table = Table(data, colWidths=[doc_width / col_count] * col_count, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174a7c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b9c4d0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fbfcfd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f7fa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def make_image(path_text: str, alt_text: str, doc_width: float, doc_height: float, styles):
    img_path = (ROOT / path_text).resolve()
    if not img_path.exists():
        return Paragraph(f"Image introuvable : {esc(path_text)}", styles["body"])

    with PILImage.open(img_path) as im:
        width, height = im.size

    # Give images a large readable area. Landscape charts get almost full width;
    # tall captures are constrained by page height.
    max_width = doc_width
    max_height = doc_height - 70
    scale = min(max_width / width, max_height / height, 1.35)
    draw_width = width * scale
    draw_height = height * scale

    caption = Paragraph(inline_markdown(alt_text), styles["caption"])
    image = Image(str(img_path), width=draw_width, height=draw_height)
    return KeepTogether([caption, image, Spacer(1, 8)])


def add_title_page(story, styles):
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Rapport global du projet SIRCH", styles["title"]))
    story.append(
        Paragraph(
            "Systeme Intelligent de Reconnaissance de Comportements violents en temps reel",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Synthese complete des manipulations, resultats, figures, courbes et captures", styles["subtitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Date : 2026-06-23", styles["subtitle"]))
    story.append(Paragraph(f"Dossier projet : {html.escape(str(ROOT))}", styles["subtitle"]))
    story.append(PageBreak())


def markdown_to_story(markdown: str, doc_width: float, doc_height: float):
    styles = make_styles()
    story = []
    add_title_page(story, styles)

    lines = markdown.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []
    paragraph_lines: list[str] = []
    first_h1_seen = False
    in_gallery_section = False
    pending_image_heading: str | None = None

    def flush_paragraph():
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines if line.strip())
            if text:
                story.append(Paragraph(inline_markdown(text), styles["body"]))
            paragraph_lines.clear()

    def flush_table():
        if table_lines:
            story.append(parse_table(table_lines, doc_width, styles))
            story.append(Spacer(1, 7))
            table_lines.clear()

    def flush_code():
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            code_lines.clear()

    while i < len(lines):
        raw = lines[i].rstrip()

        if raw.strip().startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", raw.strip())
        if image_match:
            flush_paragraph()
            flush_table()
            alt, path = image_match.groups()
            if pending_image_heading:
                alt = pending_image_heading
                pending_image_heading = None
            story.append(make_image(path, alt, doc_width, doc_height, styles))
            i += 1
            continue

        if raw.strip().startswith("|") and raw.strip().endswith("|"):
            flush_paragraph()
            table_lines.append(raw)
            i += 1
            continue

        flush_table()

        stripped = raw.strip()
        if not stripped:
            flush_paragraph()
            if not pending_image_heading:
                story.append(Spacer(1, 3))
            i += 1
            continue

        if stripped == "---":
            flush_paragraph()
            story.append(Spacer(1, 5))
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2)
            if level == 1:
                if first_h1_seen:
                    story.append(PageBreak())
                first_h1_seen = True
                story.append(Paragraph(inline_markdown(text), styles["h1"]))
            elif level == 2:
                in_gallery_section = text.startswith("19.")
                pending_image_heading = None
                if text.startswith("19.") or text.startswith("20."):
                    story.append(PageBreak())
                story.append(Paragraph(inline_markdown(text), styles["h2"]))
            elif level == 3:
                pending_image_heading = None
                story.append(Paragraph(inline_markdown(text), styles["h3"]))
            else:
                if in_gallery_section:
                    pending_image_heading = text
                    i += 1
                    continue
                story.append(Paragraph(inline_markdown(text), styles["h4"]))
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("- " + inline_markdown(stripped[2:]), styles["bullet"]))
            i += 1
            continue

        paragraph_lines.append(raw)
        i += 1

    flush_paragraph()
    flush_table()
    flush_code()
    return story


def on_page(canvas, doc):
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.drawString(doc.leftMargin, 0.55 * cm, "SIRCH - Rapport global complet")
    canvas.drawRightString(width - doc.rightMargin, 0.55 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = SOURCE_MD.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(A4),
        leftMargin=1.05 * cm,
        rightMargin=1.05 * cm,
        topMargin=0.9 * cm,
        bottomMargin=1.05 * cm,
        title="Rapport global complet du projet SIRCH",
        author="SIRCH / Codex",
    )
    story = markdown_to_story(markdown, doc.width, doc.height)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
