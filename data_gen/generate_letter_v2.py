"""
generate_letter_v2.py
-------------------
Recreates the layout of a Cambodian-style official letter (letterhead,
title, justified body paragraphs, signature block, date/org closing line,
official stamp) as a PNG image, using Pillow.

This is a re-calibrated version of generate_letter.py: every margin/gap
constant below was derived from a pixel-level analysis of
example_images/image7.jpg (row/column dark-pixel projection, cross-checked
against direct crops of each detected text band to confirm content, then
converted to fractions of page width/height so the result is
resolution-independent). See the comments next to each constant for the
measured target fraction.

Fonts used (downloaded automatically from the Google Fonts GitHub repo
if not already present in ./fonts/letterhead/):
  - Noto Sans Khmer   -> body text
  - Moul              -> bold display headings (Khmer OS "Muol" style)
  - Dancing Script    -> cursive signature flourish

Layout-recognition output: build_letter() tracks a bounding box for every
region it draws (title, header, paragraph, line, signature, stamp -- the
same LAYOUT_CLASSES taxonomy as config.py, so a detector trained on one can
be fine-tuned/evaluated on the other) and write_sample() dumps them as
YOLO-format .txt labels alongside each image, exactly like
generate_pages.py does for the synthetic page dataset.

Run:
    python -m data_gen.generate_letter_v2 --num-samples 20

Output:
    data_gen/samples/letterhead_dataset/images/letter_00000.jpg ...
    data_gen/samples/letterhead_dataset/labels/letter_00000.txt ...
        (YOLO xc,yc,w,h, normalized)
    data_gen/samples/letterhead_dataset/dataset.yaml
        (ready for `yolo detect train`)
"""

import argparse
import math
import random
import urllib.request
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from .config import LAYOUT_CLASSES
from .utils.bbox import union, xyxy_to_yolo

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

# Reference (image7.jpg) measured left/right text margin as a fraction of
# page width: ~0.062 (office-block and body-paragraph left edge both sat at
# that fraction, body's right-justified edge at 1-0.063 -- symmetric).
MARGIN_X = int(0.062 * PAGE_W)          # ~77px -- was 0.9in/135px, too wide
# Reference top margin (page top to crown-title ink top): frac 0.0382 of H.
MARGIN_TOP = int(0.0382 * PAGE_H)       # ~67px

INK = (26, 26, 26)
PAPER = (253, 253, 251)
SIGNATURE_BLUE = (26, 63, 174)
STAMP_COLOR = (178, 34, 34)


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
    except the last line which stays left-aligned. Returns
    (y_after_paragraph, line_boxes) -- line_boxes is one (x0,y0,x1,y1) box
    per wrapped line, for layout-detector "line"/"paragraph" labels."""
    lines = wrap_by_width(draw, text, font, max_width - indent_first)
    line_boxes = []
    for i, line in enumerate(lines):
        line_x = x + (indent_first if i == 0 else 0)
        is_last = (i == len(lines) - 1)
        words = line.split(" ")

        if is_last or len(words) == 1:
            draw.text((line_x, y), line, font=font, fill=color)
            line_boxes.append(draw.textbbox((line_x, y), line, font=font))
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
            # justified line spans the full available width; use textbbox
            # of the unjustified string only for its vertical (ascent/
            # descent) extent, not its (narrower) natural width.
            _, top, _, bottom = draw.textbbox((line_x, y), line, font=font)
            line_boxes.append((line_x, top, line_x + available, bottom))
        y += line_height
    return y, line_boxes


def draw_stamp(img, center, r, color=STAMP_COLOR):
    """Draws a translucent circular seal (rings + radiating ticks), the way
    a real ink stamp typically overlaps a signature. Deliberately generic --
    not a reproduction of any specific organization's real seal/text.
    """
    cx, cy = center
    alpha = 130

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color + (alpha,), width=4)
    d.ellipse((cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75),
              outline=color + (alpha,), width=2)
    for i in range(16):
        ang = 2 * math.pi * i / 16
        x_in = cx + r * 0.9 * math.cos(ang)
        y_in = cy + r * 0.9 * math.sin(ang)
        x_out = cx + r * math.cos(ang)
        y_out = cy + r * math.sin(ang)
        d.line((x_in, y_in, x_out, y_out), fill=color + (alpha,), width=2)
    d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color + (alpha,))
    img.paste(overlay, (0, 0), overlay)
    return (cx - r, cy - r, cx + r, cy + r)


# ---------------------------------------------------------------------------
# 3b. Meaningless placeholder text ("Khmer lorem ipsum")
# ---------------------------------------------------------------------------

# Letterhead/closing placeholders -- randomised per sample so a layout
# detector trained on this data doesn't just memorize one fixed string per
# region. The crown title (title1/title2) is left out of this: it's the
# fixed national letterhead motto, not something that varies per letter.
ORG_NAMES = [
    "ក្រសួងអប់រំ យុវជន និងកីឡា", "ក្រសួងសុខាភិបាល", "ក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ",
    "រដ្ឋបាលខេត្តកំពង់ចាម", "រដ្ឋបាលរាជធានីភ្នំពេញ", "អគ្គនាយកដ្ឋានពន្ធដារ",
    "ក្រសួងកសិកម្ម រុក្ខាប្រមាញ់ និងនេសាទ", "ក្រុមប្រឹក្សាឃុំសង្កាត់",
]
DEPT_NAMES = [
    "នាយកដ្ឋានឯកសារផ្លូវការ", "ការិយាល័យរដ្ឋបាល", "នាយកដ្ឋានធនធានមនុស្ស",
    "ការិយាល័យទំនាក់ទំនងសាធារណៈ", "នាយកដ្ឋានផែនការ និងហិរញ្ញវត្ថុ",
    "ការិយាល័យសវនកម្មផ្ទៃក្នុង",
]
DOC_TITLES = [
    "លិខិតជូនដំណឹងគំរូ", "លិខិតបញ្ជាក់", "សេចក្ដីជូនដំណឹង", "លិខិតអញ្ជើញ",
    "របាយការណ៍សង្ខេប", "សេចក្ដីសម្រេច", "លិខិតស្នើសុំ",
]
SIGNER_ORGS = [
    "ឈ្មោះស្ថាប័នចុះហត្ថលេខា", "នាយកប្រតិបត្តិ", "អគ្គលេខាធិការដ្ឋាន",
    "ប្រធាននាយកដ្ឋាន", "អធិការបតី",
]
SIGNER_NAMES = ["Sample", "Sophea", "Dara", "Vantha", "Ratana", "Sokha", "Chenda"]
KHMER_MONTHS = [
    "មករា", "កុម្ភៈ", "មិនា", "មេសា", "ឧសភា", "មិថុនា",
    "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ",
]


def to_khmer_numeral(n):
    """ASCII int/str -> Khmer digit string (each digit shifted into the
    Khmer numeral Unicode block, U+17E0-U+17E9)."""
    return "".join(chr(0x17E0 + int(d)) for d in str(n))


def random_date_line():
    day = to_khmer_numeral(random.randint(1, 28))
    month = random.choice(KHMER_MONTHS)
    year = to_khmer_numeral(random.randint(2023, 2026))
    return f"ថ្ងៃទី {day} ខែ{month} ឆ្នាំ{year}"


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

def build_letter(seed=7):
    """Returns {"image": PIL.Image, "detections": [(class_name, box), ...]}
    where box is (x0, y0, x1, y1) in pixel coordinates and class_name is one
    of LAYOUT_CLASSES."""
    ensure_fonts()
    if seed is not None:
        random.seed(seed)

    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    draw = ImageDraw.Draw(img)
    detections = []

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
    # image7.jpg: title1+title2 sit almost back-to-back (merge into one
    # continuous ink band at this resolution) -- gap frac ~0.0016*H.
    title1 = "ព្រះរាជាណាចក្រកម្ពុជា"
    title2 = "ជាតិ សាសនា ព្រះមហាក្សត្រ"

    w1 = text_width(draw, title1, f_crown_title)
    box1 = draw.textbbox(((PAGE_W - w1) / 2, y), title1, font=f_crown_title)
    draw.text(((PAGE_W - w1) / 2, y), title1, font=f_crown_title, fill=INK)
    y += 65

    w2 = text_width(draw, title2, f_crown_motto)
    box2 = draw.textbbox(((PAGE_W - w2) / 2, y), title2, font=f_crown_motto)
    draw.text(((PAGE_W - w2) / 2, y), title2, font=f_crown_motto, fill=INK)
    y += 62
    detections.append(("title", union([box1, box2])))

    # small centered divider rule (frac width ~0.044 of page width)
    rule_w = int(0.044 * PAGE_W)
    draw.line(
        [((PAGE_W - rule_w) / 2, y + 8), ((PAGE_W + rule_w) / 2, y + 8)],
        fill=INK, width=2
    )
    y += 29

    # ---- Office block (left-aligned), below the crown title/rule ----
    # image7.jpg: office block's two lines are nearly single-spaced
    # (gap frac ~0.0036*H), then a bigger gap (frac ~0.0327*H) to doctitle.
    org_name = random.choice(ORG_NAMES)
    dept_name = random.choice(DEPT_NAMES)
    office1_box = draw.textbbox((MARGIN_X, y), org_name, font=f_office)
    draw.text((MARGIN_X, y), org_name, font=f_office, fill=INK)
    office2_box = draw.textbbox((MARGIN_X, y + 31), dept_name, font=f_office)
    draw.text((MARGIN_X, y + 31), dept_name, font=f_office, fill=INK)
    detections.append(("header", union([office1_box, office2_box])))
    y += 112

    # ---- Document title ----
    doc_title = random.choice(DOC_TITLES)
    w_dt = text_width(draw, doc_title, f_doc_title)
    doctitle_box = draw.textbbox(((PAGE_W - w_dt) / 2, y), doc_title, font=f_doc_title)
    draw.text(((PAGE_W - w_dt) / 2, y), doc_title, font=f_doc_title, fill=INK)
    detections.append(("title", doctitle_box))
    y += 87

    # ---- Body paragraphs (randomised, diverse lengths, no real meaning) ----
    line_height = 37

    # 6 paragraphs: word counts are randomised per-paragraph so some come
    # out as a single short line and others wrap across several lines.
    # One paragraph (index 2) is rendered bold, like the emphasis line in
    # the original document. build_letter()'s `seed` makes each generated
    # sample reproducible yet distinct (seed=None gives a fresh layout).
    raw_paragraphs = generate_paragraphs(
        count=6,
        seed=None,  # already seeded once at the top of build_letter()
        min_words=6,
        max_words=55,
        bold_indices={2},
    )

    # image7.jpg: first-line indent measured at frac ~0.066*W beyond the
    # base left margin (i.e. indent_first ~= 0.066*PAGE_W).
    indent_first = int(0.066 * PAGE_W)

    for text, is_bold, indent in raw_paragraphs:
        font = f_body_bold if is_bold else f_body
        y, line_boxes = draw_justified_paragraph(
            draw, text, font,
            x=MARGIN_X, y=y, max_width=content_w,
            line_height=line_height,
            indent_first=indent_first if indent else 0,
        )
        for box in line_boxes:
            detections.append(("line", box))
        if line_boxes:
            detections.append(("paragraph", union(line_boxes)))
        y += 25  # paragraph gap (frac ~0.0145*H)

    y += 25

    # ---- Closing: date + org name, then signature (right-aligned) ----
    date_line = random_date_line()
    w_date = text_width(draw, date_line, f_closing)
    date_box = draw.textbbox((PAGE_W - MARGIN_X - w_date, y), date_line, font=f_closing)
    draw.text((PAGE_W - MARGIN_X - w_date, y), date_line,
              font=f_closing, fill=INK)
    detections.append(("line", date_box))
    y += 37

    org_line = random.choice(SIGNER_ORGS)
    w_org = text_width(draw, org_line, f_closing_bold)
    org_box = draw.textbbox((PAGE_W - MARGIN_X - w_org, y), org_line, font=f_closing_bold)
    draw.text((PAGE_W - MARGIN_X - w_org, y), org_line,
              font=f_closing_bold, fill=INK)
    detections.append(("line", org_box))
    y += 30

    # ---- Official stamp, overlapping the signature area (generic seal --
    # see draw_stamp docstring) ----
    stamp_box = draw_stamp(img, center=(PAGE_W - MARGIN_X - 110, y + 55), r=68)
    detections.append(("stamp", stamp_box))

    # ---- Signature flourish (right-aligned, below the org name) ----
    flourish = random.choice(SIGNER_NAMES)
    w_fl = text_width(draw, flourish, f_flourish)
    flourish_box = draw.textbbox((PAGE_W - MARGIN_X - w_fl - 40, y), flourish, font=f_flourish)
    draw.text((PAGE_W - MARGIN_X - w_fl - 40, y), flourish,
              font=f_flourish, fill=SIGNATURE_BLUE)
    detections.append(("signature", flourish_box))

    return {"image": img, "detections": detections}


# ---------------------------------------------------------------------------
# 5. Dataset output (YOLO labels + dataset.yaml, same convention as
#    generate_pages.py)
# ---------------------------------------------------------------------------

def write_sample(out_dir: Path, name: str, sample: dict):
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img = sample["image"]
    img_w, img_h = img.size
    img.save(images_dir / f"{name}.jpg", quality=92, dpi=(DPI, DPI))

    label_lines = []
    for cls_name, box in sample["detections"]:
        cls_id = LAYOUT_CLASSES.index(cls_name)
        xc, yc, w, h = xyxy_to_yolo(box, img_w, img_h)
        label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    (labels_dir / f"{name}.txt").write_text("\n".join(label_lines), encoding="utf-8")


def write_dataset_yaml(out_dir: Path):
    data = {
        "path": str(out_dir.resolve()),
        "train": "images",
        "val": "images",
        "names": {i: name for i, name in enumerate(LAYOUT_CLASSES)},
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "samples" / "letterhead_dataset")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        sample = build_letter(seed=args.seed + i)
        write_sample(args.out_dir, f"letter_{i:05d}", sample)
    write_dataset_yaml(args.out_dir)
    print(f"Wrote {args.num_samples} samples to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
