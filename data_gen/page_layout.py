"""Page layout templates: define where titles/headers/paragraphs go on a page,
before any text is rendered into them. Each template returns a list of region
dicts: {"type": "title"|"header"|"paragraph"|"field"|"signature"|"photo",
"box": (x0, y0, x1, y1), "n_lines": int (paragraphs only)}. `page_renderer.py`
fills these regions with actual rendered text (or, for "signature"/"photo",
placeholder graphics) and produces per-line boxes for the "line" class.

Templates: book, research_paper, national_id (dense field lists, A4),
id_card_front / id_card_back (small ID-1 card canvas, modeled on real Khmer
ID cards), birth_certificate (label/value grid form with signature blocks),
official_letter (letterhead + centered title + body + signature block,
modeled on real government notices / press releases).
"""
import random

from .config import PageRenderConfig


def book_page(cfg: PageRenderConfig):
    w, h = cfg.page_size
    m = cfg.margin
    regions = []
    y = m
    y += int(cfg.title_font_size[1] * cfg.line_spacing) + 20
    regions.append({"type": "header", "box": (m, m, w - m, y)})

    n_paragraphs = random.randint(2, 4)
    content_bottom = h - m
    para_height = (content_bottom - y) // n_paragraphs
    for i in range(n_paragraphs):
        top = y + i * para_height + 10
        bottom = y + (i + 1) * para_height - 10
        n_lines = max(2, int((bottom - top) // (cfg.paragraph_font_size[1] * cfg.line_spacing)))
        regions.append({"type": "paragraph", "box": (m, top, w - m, bottom), "n_lines": n_lines})
    return regions


def research_paper(cfg: PageRenderConfig):
    w, h = cfg.page_size
    m = cfg.margin
    regions = []
    y = m
    title_bottom = y + int(cfg.title_font_size[1] * cfg.line_spacing) + 16
    regions.append({"type": "title", "box": (m, y, w - m, title_bottom)})

    y = title_bottom + 12
    authors_bottom = y + int(cfg.header_font_size[1] * cfg.line_spacing)
    regions.append({"type": "header", "box": (m, y, w - m, authors_bottom)})

    y = authors_bottom + 20
    abstract_header_bottom = y + int(cfg.header_font_size[1] * cfg.line_spacing)
    regions.append({"type": "header", "box": (m, y, w - m, abstract_header_bottom)})

    y = abstract_header_bottom + 8
    abstract_bottom = y + 5 * cfg.paragraph_font_size[1] * cfg.line_spacing
    regions.append({"type": "paragraph", "box": (m, y, w - m, abstract_bottom), "n_lines": 5})

    y = int(abstract_bottom) + cfg.paragraph_gap
    n_sections = random.randint(2, 3)
    section_height = (h - m - y) // n_sections
    for i in range(n_sections):
        top = y + i * section_height
        header_bottom = top + int(cfg.header_font_size[1] * cfg.line_spacing)
        regions.append({"type": "header", "box": (m, top, w - m, header_bottom)})
        body_top = header_bottom + 6
        body_bottom = top + section_height - 10
        n_lines = max(2, int((body_bottom - body_top) // (cfg.paragraph_font_size[1] * cfg.line_spacing)))
        regions.append({"type": "paragraph", "box": (m, body_top, w - m, body_bottom), "n_lines": n_lines})
    return regions


def national_id_document(cfg: PageRenderConfig):
    """Loosely mimics a national document: a title/header block up top,
    then a series of short label+value lines (rendered as many small
    single-line 'paragraph' regions rather than dense text blocks).
    """
    w, h = cfg.page_size
    m = cfg.margin
    regions = []
    y = m
    title_bottom = y + int(cfg.title_font_size[1] * cfg.line_spacing) + 16
    regions.append({"type": "title", "box": (m, y, w - m, title_bottom)})

    y = title_bottom + 10
    header_bottom = y + int(cfg.header_font_size[1] * cfg.line_spacing)
    regions.append({"type": "header", "box": (m, y, w - m, header_bottom)})

    y = header_bottom + 30
    n_fields = random.randint(6, 10)
    field_height = (h - m - y) // n_fields
    for i in range(n_fields):
        top = y + i * field_height
        bottom = top + field_height - 8
        regions.append({"type": "paragraph", "box": (m, top, w - m, bottom), "n_lines": 1})
    return regions


def id_card_front(cfg: PageRenderConfig):
    """Modeled on a real Khmer national ID card: header block top, a photo
    slot on the left, field rows to its right, and a signature at bottom
    right. Uses `id_card_size` instead of the A4 `page_size`.
    """
    w, h = cfg.id_card_size
    m = 24
    regions = []

    header_bottom = m + int(cfg.header_font_size[1] * cfg.line_spacing) * 2
    regions.append({"type": "header", "box": (m, m, w - m, header_bottom)})

    photo_w, photo_h = int(w * 0.22), int(h * 0.55)
    photo_top = header_bottom + 14
    regions.append({"type": "photo", "box": (m, photo_top, m + photo_w, photo_top + photo_h)})

    field_left = m + photo_w + 20
    n_fields = 5
    field_area_bottom = h - m - 60
    field_height = (field_area_bottom - photo_top) // n_fields
    for i in range(n_fields):
        top = photo_top + i * field_height
        bottom = top + field_height - 4
        regions.append({"type": "field", "box": (field_left, top, w - m, bottom)})

    regions.append({"type": "signature", "box": (w - m - 180, h - m - 55, w - m, h - m)})
    return regions


def id_card_back(cfg: PageRenderConfig):
    """Back of an ID card: a couple of Khmer field rows on top, then a
    monospace-looking MRZ block at the bottom (rendered as 'line' rows).
    """
    w, h = cfg.id_card_size
    m = 24
    regions = []

    n_fields = 3
    field_top = m
    field_height = int(h * 0.12)
    for i in range(n_fields):
        top = field_top + i * field_height
        regions.append({"type": "field", "box": (m, top, w - m, top + field_height - 6)})

    mrz_top = field_top + n_fields * field_height + 20
    mrz_line_h = int(h * 0.11)
    for i in range(3):
        top = mrz_top + i * mrz_line_h
        regions.append({"type": "paragraph", "box": (m, top, w - m, top + mrz_line_h - 4), "n_lines": 1})
    return regions


def birth_certificate(cfg: PageRenderConfig):
    """Modeled on a real extract-of-birth-certificate form: title block,
    then a grid of label/value field rows, then two signature blocks
    (registrar + local authority) side by side at the bottom.
    """
    w, h = cfg.page_size
    m = cfg.margin
    regions = []
    y = m

    header_bottom = y + int(cfg.header_font_size[1] * cfg.line_spacing)
    regions.append({"type": "header", "box": (m, y, w - m, header_bottom)})

    y = header_bottom + 16
    title_bottom = y + int(cfg.title_font_size[1] * cfg.line_spacing)
    regions.append({"type": "title", "box": (m, y, w - m, title_bottom)})

    y = title_bottom + 30
    n_fields = 10
    signature_zone_h = 140
    field_area_bottom = h - m - signature_zone_h
    field_height = (field_area_bottom - y) // n_fields
    for i in range(n_fields):
        top = y + i * field_height
        bottom = top + field_height - 6
        regions.append({"type": "field", "box": (m, top, w - m, bottom)})

    sig_top = field_area_bottom + 40
    col_w = (w - 2 * m) // 2
    regions.append({"type": "signature", "box": (m + 20, sig_top, m + col_w - 20, sig_top + 90)})
    regions.append({"type": "signature", "box": (m + col_w + 20, sig_top, w - m - 20, sig_top + 90)})
    return regions


def official_letter(cfg: PageRenderConfig):
    """Modeled on a government notice / press release: letterhead (header)
    at top, a centered title, body paragraphs, and a signature block at the
    bottom right.
    """
    w, h = cfg.page_size
    m = cfg.margin
    regions = []
    y = m

    header_bottom = y + int(cfg.header_font_size[1] * cfg.line_spacing) * 2
    regions.append({"type": "header", "box": (m, y, w - m, header_bottom)})

    y = header_bottom + 20
    title_bottom = y + int(cfg.title_font_size[1] * cfg.line_spacing)
    regions.append({"type": "title", "box": (m, y, w - m, title_bottom)})

    y = title_bottom + 24
    signature_zone_h = 130
    body_bottom = h - m - signature_zone_h
    n_paragraphs = random.randint(2, 3)
    para_height = (body_bottom - y) // n_paragraphs
    for i in range(n_paragraphs):
        top = y + i * para_height + 8
        bottom = y + (i + 1) * para_height - 8
        n_lines = max(2, int((bottom - top) // (cfg.paragraph_font_size[1] * cfg.line_spacing)))
        regions.append({"type": "paragraph", "box": (m, top, w - m, bottom), "n_lines": n_lines})

    sig_top = body_bottom + 30
    regions.append({"type": "signature", "box": (w - m - 220, sig_top, w - m, sig_top + 90)})
    return regions


# name -> (builder, size attribute on PageRenderConfig to use as the canvas size)
TEMPLATES = {
    "book": (book_page, "page_size"),
    "research_paper": (research_paper, "page_size"),
    "national_id": (national_id_document, "page_size"),
    "id_card_front": (id_card_front, "id_card_size"),
    "id_card_back": (id_card_back, "id_card_size"),
    "birth_certificate": (birth_certificate, "page_size"),
    "official_letter": (official_letter, "page_size"),
}


def sample_layout(cfg: PageRenderConfig, template: str = None):
    name = template or random.choice(list(TEMPLATES))
    builder, size_attr = TEMPLATES[name]
    canvas_size = getattr(cfg, size_attr)
    return name, builder(cfg), canvas_size
