"""
generate_letter.py
-------------------
Recreates the layout of a Cambodian-style official letter (letterhead,
title, justified body paragraphs, signature block, date/org closing line)
as a PNG image, using Pillow. Superseded by generate_letter_v2.py (which
recalibrated every spacing constant against a real reference photo and adds
layout-detector labels) -- this version is kept as the simpler, un-tuned
baseline it was built from.

Fonts used (downloaded automatically from the Google Fonts GitHub repo
if not already present in ./fonts/letterhead/):
  - Noto Sans Khmer   -> body text
  - Moul              -> bold display headings (Khmer OS "Muol" style)
  - Dancing Script    -> cursive signature flourish

Run:
    python -m data_gen.generate_letter

Output:
    data_gen/samples/letterhead/khmer_letter.png
"""

import os
import random
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 1. Fonts: download once into fonts/letterhead/ if missing
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_DIR = SCRIPT_DIR / "fonts" / "letterhead"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONT_SOURCES = {
    "NotoSansKhmer.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "notosanskhmer/NotoSansKhmer%5Bwdth%2Cwght%5D.ttf"
    ),
    "Moul.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "moul/Moul-Regular.ttf"
    ),
    "DancingScript.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "dancingscript/DancingScript%5Bwght%5D.ttf"
    ),
}


def ensure_fonts():
    for filename, url in FONT_SOURCES.items():
        path = FONT_DIR / filename
        if not path.exists():
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, path)


def load_font(name, size, variation_weight=None):
    """Load a font at a given pixel size. Optionally set a variable-font
    weight axis (e.g. 700 for bold) if the font supports it."""
    font = ImageFont.truetype(str(FONT_DIR / name), size)
    if variation_weight is not None:
        try:
            font.set_variation_by_axes([variation_weight])
        except Exception:
            pass  # font isn't variable, or has no weight axis -- ignore
    return font


# ---------------------------------------------------------------------------
# 2. Page setup
# ---------------------------------------------------------------------------

# A4 at 150 DPI
DPI = 150
PAGE_W = int(8.27 * DPI)
PAGE_H = int(11.69 * DPI)
MARGIN_X = int(0.9 * DPI)
MARGIN_TOP = int(0.36 * DPI)

INK = (26, 26, 26)
PAPER = (253, 253, 251)
SIGNATURE_BLUE = (26, 63, 174)


# ---------------------------------------------------------------------------
# 3. Text-layout helpers
# ---------------------------------------------------------------------------

def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_by_width(draw, text, font, max_width):
    """Word-wrap text (splitting on spaces) so each line fits max_width."""
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        trial = (current + " " + word).strip()
        if text_width(draw, trial, font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_justified_paragraph(draw, text, font, x, y, max_width,
                              line_height, indent_first=0, color=INK):
    """Draw a paragraph, justified (extra space distributed between words),
    except the last line which stays left-aligned. Returns the y position
    after the paragraph."""
    lines = wrap_by_width(draw, text, font, max_width - indent_first)
    for i, line in enumerate(lines):
        line_x = x + (indent_first if i == 0 else 0)
        is_last = (i == len(lines) - 1)
        words = line.split(" ")

        if is_last or len(words) == 1:
            draw.text((line_x, y), line, font=font, fill=color)
        else:
            # Justify: distribute leftover space evenly between words
            words_width = sum(text_width(draw, w, font) for w in words)
            gap_count = len(words) - 1
            available = max_width - (indent_first if i == 0 else 0)
            gap = (available - words_width) / gap_count if gap_count else 0
            cx = line_x
            for w in words:
                draw.text((cx, y), w, font=font, fill=color)
                cx += text_width(draw, w, font) + gap
        y += line_height
    return y


# ---------------------------------------------------------------------------
# 3b. Meaningless placeholder text ("Khmer lorem ipsum")
# ---------------------------------------------------------------------------

# A pool of real Khmer syllables/short words, recombined at random. The
# output does not form coherent sentences -- it's filler text only, used
# purely to test layout with varying line/paragraph lengths.
_WORD_BANK = [
    "កថា", "ខណ្ឌ", "ឯកសារ", "គំរូ", "អក្សរ", "បន្ទាត់", "ចន្លោះ", "ទំហំ",
    "ព័ត៌មាន", "ការិយាល័យ", "ស្ថាប័ន", "ផ្លូវការ", "ឃ្លា", "ប្លង់", "ទំព័រ",
    "រចនាបថ", "សំណុំ", "កំណត់ត្រា", "លិខិត", "ដំណឹង", "ខ្លឹមសារ", "ទម្រង់",
    "ភាសា", "សញ្ញា", "សំណួរ", "ចម្លើយ", "ចំណុច", "ព្រាង", "កំណែ", "ច្បាប់ចម្លង",
    "ការណ៍", "លទ្ធផល", "សេចក្ដី", "ព័ត៌មានបន្ថែម", "ការចុះឈ្មោះ", "ការអនុម័ត",
    "ការត្រួតពិនិត្យ", "ការផ្ទៀងផ្ទាត់", "ការចេញផ្សាយ", "កាលបរិច្ឆេទ",
]


def random_khmer_words(n):
    return " ".join(random.choice(_WORD_BANK) for _ in range(n))


def generate_paragraph(min_words=8, max_words=70):
    """Return one nonsense paragraph with a randomly chosen word count,
    so consecutive paragraphs naturally vary in length."""
    n = random.randint(min_words, max_words)
    text = random_khmer_words(n)
    return text + "។"


def generate_paragraphs(count=5, seed=None,
                         min_words=8, max_words=70,
                         bold_indices=None):
    """Build a list of (text, is_bold, indent) tuples with diverse,
    randomised lengths -- some short one-liners, some long multi-line
    paragraphs. Pass a seed for reproducible output."""
    if seed is not None:
        random.seed(seed)
    bold_indices = bold_indices or set()

    paragraphs = []
    for i in range(count):
        text = generate_paragraph(min_words, max_words)
        is_bold = i in bold_indices
        paragraphs.append((text, is_bold, True))
    return paragraphs


# ---------------------------------------------------------------------------
# 4. Build the page
# ---------------------------------------------------------------------------

def build_letter():
    ensure_fonts()

    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    draw = ImageDraw.Draw(img)

    content_w = PAGE_W - 2 * MARGIN_X

    # ---- Fonts ----
    f_crown_title = load_font("Moul.ttf", 34)
    f_crown_motto = load_font("Moul.ttf", 26)
    f_office = load_font("NotoSansKhmer.ttf", 20, variation_weight=600)
    f_doc_title = load_font("NotoSansKhmer.ttf", 26, variation_weight=700)
    f_body = load_font("NotoSansKhmer.ttf", 21)
    f_body_bold = load_font("NotoSansKhmer.ttf", 21, variation_weight=700)
    f_flourish = load_font("DancingScript.ttf", 44, variation_weight=600)
    f_closing = load_font("NotoSansKhmer.ttf", 20)
    f_closing_bold = load_font("NotoSansKhmer.ttf", 21, variation_weight=700)

    y = MARGIN_TOP

    # ---- Crown title (centered) ----
    title1 = "ព្រះរាជាណាចក្រកម្ពុជា"
    title2 = "ជាតិ សាសនា ព្រះមហាក្សត្រ"

    w1 = text_width(draw, title1, f_crown_title)
    draw.text(((PAGE_W - w1) / 2, y), title1, font=f_crown_title, fill=INK)
    y += 81

    w2 = text_width(draw, title2, f_crown_motto)
    draw.text(((PAGE_W - w2) / 2, y), title2, font=f_crown_motto, fill=INK)
    y += 90

    # small centered divider rule under the motto
    rule_w = 90
    draw.line(
        [((PAGE_W - rule_w) / 2, y + 8), ((PAGE_W + rule_w) / 2, y + 8)],
        fill=INK, width=2
    )
    y += 62

    # ---- Office block (left-aligned), below the crown title/rule ----
    draw.text((MARGIN_X, y), "ឈ្មោះស្ថាប័ន", font=f_office, fill=INK)
    draw.text((MARGIN_X, y + 35), "នាយកដ្ឋានឯកសារផ្លូវការ", font=f_office, fill=INK)
    y += 69

    # ---- Document title ----
    doc_title = "លិខិតជូនដំណឹងគំរូ"
    w_dt = text_width(draw, doc_title, f_doc_title)
    draw.text(((PAGE_W - w_dt) / 2, y), doc_title, font=f_doc_title, fill=INK)
    y += 58

    # ---- Body paragraphs (randomised, diverse lengths, no real meaning) ----
    line_height = 38

    # 6 paragraphs: word counts are randomised per-paragraph so some come
    # out as a single short line and others wrap across several lines.
    # One paragraph (index 2) is rendered bold, like the emphasis line in
    # the original document. Pass seed=<int> for reproducible output, or
    # seed=None (default) for a fresh random layout every run.
    raw_paragraphs = generate_paragraphs(
        count=6,
        seed=7,
        min_words=6,
        max_words=55,
        bold_indices={2},
    )

    for text, is_bold, indent in raw_paragraphs:
        font = f_body_bold if is_bold else f_body
        y = draw_justified_paragraph(
            draw, text, font,
            x=MARGIN_X, y=y, max_width=content_w,
            line_height=line_height,
            indent_first=44 if indent else 0,
        )
        y += 25  # paragraph gap

    y += 25

    # ---- Closing: date + org name, then signature (right-aligned) ----
    date_line = "ថ្ងៃទី ១៥ ខែកក្កដា ឆ្នាំ២០២៦"
    w_date = text_width(draw, date_line, f_closing)
    draw.text((PAGE_W - MARGIN_X - w_date, y), date_line,
              font=f_closing, fill=INK)
    y += 37

    org_line = "ឈ្មោះស្ថាប័នចុះហត្ថលេខា"
    w_org = text_width(draw, org_line, f_closing_bold)
    draw.text((PAGE_W - MARGIN_X - w_org, y), org_line,
              font=f_closing_bold, fill=INK)
    y += 28

    # ---- Signature flourish (right-aligned, below the org name) ----
    flourish = "Sample"
    w_fl = text_width(draw, flourish, f_flourish)
    draw.text((PAGE_W - MARGIN_X - w_fl - 40, y), flourish,
              font=f_flourish, fill=SIGNATURE_BLUE)

    return img


if __name__ == "__main__":
    letter = build_letter()
    out_dir = SCRIPT_DIR / "samples" / "letterhead"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "khmer_letter.png"
    letter.save(out_path, dpi=(DPI, DPI))
    print(f"Saved: {out_path}")
