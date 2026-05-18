"""
make_pptx.py  —  Convert slides.md → CineAI_Slides.pptx
Run: python docs/make_pptx.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import re, os

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = RGBColor(0x0f, 0x0f, 0x1a)   # near-black navy
ACCENT   = RGBColor(0x7c, 0x3a, 0xed)   # purple
ACCENT2  = RGBColor(0x3b, 0x82, 0xf6)   # blue
WHITE    = RGBColor(0xff, 0xff, 0xff)
GRAY     = RGBColor(0xcc, 0xcc, 0xdd)
GREEN    = RGBColor(0x10, 0xb9, 0x81)

W, H = Inches(13.33), Inches(7.5)       # 16:9


def set_bg(slide, prs):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def add_textbox(slide, text, x, y, w, h,
                size=20, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    return txb


def accent_bar(slide, y=Inches(0.18)):
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), y, W, Inches(0.07)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


def parse_slides(md_path):
    """Split markdown on '---' separators; return list of raw slide strings."""
    text = open(md_path, encoding="utf-8").read()
    # Remove <!-- slide N --> comments
    text = re.sub(r"<!--.*?-->", "", text)
    parts = re.split(r"\n---\n", text)
    return [p.strip() for p in parts if p.strip()]


def render_slide(prs, raw):
    slide_layout = prs.slide_layouts[6]   # blank
    slide = prs.slides.add_slide(slide_layout)
    set_bg(slide, prs)
    accent_bar(slide)

    lines = raw.splitlines()

    # --- detect title line(s) ---
    title_lines, body_lines = [], []
    for i, ln in enumerate(lines):
        if ln.startswith("# ") and not title_lines:
            title_lines.append(ln.lstrip("# ").strip())
        elif ln.startswith("## "):
            title_lines.append(ln.lstrip("# ").strip())
        elif ln.startswith("### ") and not title_lines:
            title_lines.append(ln.lstrip("# ").strip())
        else:
            body_lines.append(ln)

    title_text = "\n".join(title_lines)

    # ── Title ────────────────────────────────────────────────────────────────
    if title_text:
        is_cover = any(ln.startswith("# ") for ln in lines)
        size  = 44 if is_cover else 32
        color = WHITE if is_cover else ACCENT2
        y     = Inches(0.6) if is_cover else Inches(0.4)
        add_textbox(slide, title_text,
                    Inches(0.6), y, Inches(12), Inches(1.5),
                    size=size, bold=True, color=color, align=PP_ALIGN.LEFT)

    # ── Body ─────────────────────────────────────────────────────────────────
    body_text = "\n".join(body_lines).strip()

    # Detect table
    table_lines = [l for l in body_lines if "|" in l and not l.strip().startswith("<!--")]
    has_table   = len(table_lines) >= 2

    if has_table:
        _render_table_slide(slide, body_lines)
    else:
        _render_text_slide(slide, body_lines)

    return slide


def _render_text_slide(slide, body_lines):
    """Render bullet / code body."""
    content = []
    in_code = False
    for ln in body_lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if not ln.strip():
            continue
        # strip markdown bold/italic
        ln_clean = re.sub(r"\*\*(.*?)\*\*", r"\1", ln)
        ln_clean = re.sub(r"\*(.*?)\*",     r"\1", ln_clean)
        ln_clean = re.sub(r"`(.*?)`",       r"\1", ln_clean)
        content.append(("code" if in_code else "text", ln_clean))

    if not content:
        return

    # single text box starting at y=2.2
    txb = slide.shapes.add_textbox(Inches(0.6), Inches(2.0), Inches(12.1), Inches(5.0))
    tf  = txb.text_frame
    tf.word_wrap = True

    first = True
    for kind, text in content:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.LEFT

        # indentation for bullets
        is_bullet = text.startswith("- ") or text.startswith("* ")
        if is_bullet:
            text = "  •  " + text[2:]

        is_heading = text.startswith("### ")
        if is_heading:
            text = text[4:]

        run = p.add_run()
        run.text = text
        run.font.color.rgb = RGBColor(0xd0, 0xd0, 0xff) if kind == "code" else (
            ACCENT2 if is_heading else GRAY)
        run.font.size  = Pt(14) if kind == "code" else (Pt(18) if is_heading else Pt(16))
        run.font.bold  = is_heading


def _render_table_slide(slide, body_lines):
    """Parse first markdown table found and render as pptx table."""
    rows_data = []
    for ln in body_lines:
        if "|" not in ln:
            continue
        cells = [c.strip() for c in ln.split("|") if c.strip()]
        if all(set(c) <= set("-: ") for c in cells):
            continue   # separator row
        rows_data.append(cells)

    if not rows_data:
        return

    max_cols = max(len(r) for r in rows_data)
    n_rows   = len(rows_data)

    left, top = Inches(0.6), Inches(2.1)
    width  = Inches(12.1)
    height = Inches(min(0.45 * n_rows, 4.8))

    tbl = slide.shapes.add_table(n_rows, max_cols, left, top, width, height).table
    tbl.columns[0].width = Inches(4)

    for r_i, row in enumerate(rows_data):
        for c_i, cell_text in enumerate(row):
            if c_i >= max_cols:
                break
            cell = tbl.cell(r_i, c_i)
            cell.text = re.sub(r"\*\*(.*?)\*\*", r"\1", cell_text)
            tf = cell.text_frame
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
            run = tf.paragraphs[0].runs[0] if tf.paragraphs[0].runs else tf.paragraphs[0].add_run()
            run.font.size  = Pt(13)
            run.font.color.rgb = WHITE
            run.font.bold  = (r_i == 0)
            # header row bg
            fill = cell.fill
            fill.solid()
            fill.fore_color.rgb = ACCENT if r_i == 0 else (
                RGBColor(0x1e, 0x1e, 0x35) if r_i % 2 == 0 else RGBColor(0x17, 0x17, 0x2a))


def main():
    md_path  = os.path.join(os.path.dirname(__file__), "slides.md")
    out_path = os.path.join(os.path.dirname(__file__), "CineAI_Slides.pptx")

    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    raw_slides = parse_slides(md_path)
    print(f"[INFO] Parsed {len(raw_slides)} slides from {md_path}")

    for i, raw in enumerate(raw_slides, 1):
        render_slide(prs, raw)
        print(f"  [OK] Slide {i:02d} rendered")

    prs.save(out_path)
    print(f"\n[DONE] Saved -> {out_path}")


if __name__ == "__main__":
    main()
