"""
generate_specimen_id.py
------------------------
Builds a synthetic "SPECIMEN" Cambodian ID card using a field layout
calibrated against example_images/image-id-card-1.jpg (a real card,
photographed at an angle on a tablecloth -- so positions below are close
proportional matches, not pixel-exact, since the source has perspective
skew that a flat render can't reproduce exactly).

All data is fictional and the card is clearly marked as such (a diagonal
"SPECIMEN" watermark plus a corner disclaimer) -- this generator is for
layout/OCR training data, not for producing anything that could pass as a
real document. The photo is a generic silhouette placeholder, never a real
face; the security-pattern/hologram look is a generic recreation (rainbow-
tiled "KINGDOM OF CAMBODIA" text + an abstract translucent swirl), not a
reproduction of the real card's exact security design.

Run:
    python -m data_gen.generate_specimen_id

Output:
    data_gen/samples/specimen_id/specimen_id_card.png
"""

import math
import random
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Fonts: shared with generate_letter*.py -- download once if missing
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_DIR = SCRIPT_DIR / "fonts" / "letterhead"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONT_SOURCES = {
    "NotoSansKhmer.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "notosanskhmer/NotoSansKhmer%5Bwdth%2Cwght%5D.ttf"
    ),
    # Real ID cards print the MRZ / ID-number in a true fixed-pitch OCR-style
    # font -- NotoSansKhmer's proportional Latin glyphs can't reproduce that
    # grid alignment no matter how they're spaced (a manual equal-advance
    # hack was tried earlier and looked worse). Courier Prime is a real
    # monospace font, so draw.text() lines up every character natively.
    "CourierPrime-Regular.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "courierprime/CourierPrime-Regular.ttf"
    ),
    "CourierPrime-Bold.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "courierprime/CourierPrime-Bold.ttf"
    ),
}


def ensure_fonts():
    for filename, url in FONT_SOURCES.items():
        path = FONT_DIR / filename
        if not path.exists():
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, path)


def load_font(name, size, weight=None):
    f = ImageFont.truetype(str(FONT_DIR / name), size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


CARD_W = 878
# Standard ID-1 / CR80 card ratio is 85.60mm x 53.98mm = 1.586:1
CARD_H = round(CARD_W / 1.586)  # -> 554
# Real CR80/ID-1 cards have a ~3.18mm corner radius on an 85.6mm-wide card
# (~3.7% of width) -- the previous fixed radius=22 (~2.5%) read as barely
# rounded at all.
CARD_RADIUS = round(CARD_W * 0.037)  # -> 32
# Output is generated at the calibrated CARD_W/CARD_H above, then downscaled
# to this size when saved -- smaller final image, same layout proportions.
OUTPUT_SCALE = 0.7

_SCALE_Y = CARD_H / 567  # rescale all y-coordinates below, which were
                          # originally measured against a 567px-tall crop

BG = (223, 227, 214)      # pale sage-grey -- image-id-card-1.jpg's card body
                          # is a muted khaki/sage, not a neutral grey-white
INK = (25, 25, 25)


def _sy(y):
    return round(y * _SCALE_Y)

# ---- Layout boxes, reverse-engineered from image-id-card-1.jpg, with
# y-coordinates rescaled to the corrected card height. The MRZ rows
# (mrz1-3) are widened to run the full card width, matching how MRZ zones
# are actually printed on real ID cards (edge to edge with a small uniform
# margin), rather than the tighter text-only extent.
MRZ_MARGIN = 24
BOX = {
    "id_number":        (568, _sy(33), 794, _sy(64)),
    "photo":             (37, _sy(55), 203, _sy(256)),
    "signature":         (63, _sy(260), 174, _sy(322)),
    "name_khmer":       (225, _sy(56), 418, _sy(87)),
    "name_latin":       (360, _sy(95), 457, _sy(113)),
    "dob_sex_height":   (225, _sy(119), 732, _sy(152)),
    "place_of_birth":   (227, _sy(152), 668, _sy(184)),
    "address1":         (225, _sy(186), 498, _sy(218)),
    "address2":         (227, _sy(217), 548, _sy(248)),
    "issue_expiry":     (227, _sy(257), 645, _sy(286)),
    "marks":            (227, _sy(284), 610, _sy(320)),
    "mrz1":  (MRZ_MARGIN, _sy(384), CARD_W - MRZ_MARGIN, _sy(415)),
    "mrz2":  (MRZ_MARGIN, _sy(423), CARD_W - MRZ_MARGIN, _sy(454)),
    "mrz3":  (MRZ_MARGIN, _sy(461), CARD_W - MRZ_MARGIN, _sy(493)),
}


_MRZ_CHECK_WEIGHTS = (7, 3, 1)


def _mrz_char_value(c):
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c) - ord("A") + 10


def mrz_check_digit(s):
    """ICAO 9303 check-digit algorithm (weights 7,3,1 repeating)."""
    total = sum(_mrz_char_value(c) * _MRZ_CHECK_WEIGHTS[i % 3] for i, c in enumerate(s))
    return str(total % 10)


def build_td1_mrz(doc_number, dob_yymmdd, sex, expiry_yymmdd, nationality, surname, given_names):
    """Builds a real, ICAO 9303 TD1-format 3x30 MRZ (same layout as the
    chevron-filled rows on image-id-card-1.jpg) with correctly computed
    check digits, instead of hand-typed digits that only *looked* right.
    """
    line1 = "ID" + nationality + doc_number.ljust(9, "<")
    line1 += mrz_check_digit(doc_number.ljust(9, "<"))
    line1 = line1.ljust(30, "<")

    optional2 = "<" * 11
    line2 = dob_yymmdd + mrz_check_digit(dob_yymmdd) + sex + expiry_yymmdd
    line2 += mrz_check_digit(expiry_yymmdd) + nationality + optional2
    composite_input = (
        doc_number.ljust(9, "<") + mrz_check_digit(doc_number.ljust(9, "<"))
        + dob_yymmdd + mrz_check_digit(dob_yymmdd)
        + expiry_yymmdd + mrz_check_digit(expiry_yymmdd)
        + optional2
    )
    line2 += mrz_check_digit(composite_input)

    line3 = (surname + "<<" + given_names).ljust(30, "<")[:30]

    assert len(line1) == len(line2) == len(line3) == 30
    return line1, line2, line3


def draw_justified_mono_line(draw, box, text, font, fill):
    """Draws `text` stretched to exactly fill the box's width -- real MRZ
    rows are printed edge to edge (see image-id-card-1.jpg), not left-packed
    at the font's natural width. Safe here (unlike the proportional-font
    tracking hack tried and reverted elsewhere in this file) because `font`
    is a true monospace font: every character already has the same natural
    advance, so adding one uniform extra gap between characters keeps the
    row perfectly even instead of introducing lopsided-looking spacing.
    """
    x0, y0, x1, y1 = box
    target_w = x1 - x0
    natural_w = draw.textlength(text, font=font)
    n = len(text)
    gap = (target_w - natural_w) / (n - 1) if n > 1 else 0
    x = x0
    for ch in text:
        draw.text((x, y0), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + gap


def draw_generic_avatar(draw, img, box, seed_color):
    """Draw a simple generic silhouette placeholder -- NOT a real photo.
    Rendered on a separate layer and clipped to the box so no part of
    the shape can extend past the photo frame."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0

    # background fill for the photo box
    draw.rectangle(box, fill=(225, 228, 232), outline=(160, 160, 165), width=2)

    # draw the silhouette on its own layer sized exactly to the box,
    # then paste back in -- guarantees nothing escapes the frame
    layer = Image.new("RGB", (bw, bh), (225, 228, 232))
    ldraw = ImageDraw.Draw(layer)
    cx = bw / 2
    head_r = bw * 0.22
    head_cy = bh * 0.34
    ldraw.ellipse([cx-head_r, head_cy-head_r, cx+head_r, head_cy+head_r], fill=seed_color)
    body_w = bw * 0.66
    body_top = head_cy + head_r * 0.65
    ldraw.ellipse([cx-body_w/2, body_top, cx+body_w/2, bh + body_w*0.3], fill=seed_color)

    img.paste(layer, (x0, y0))
    draw.rectangle(box, outline=(160, 160, 165), width=2)


def draw_signature(draw, box, color=(20, 30, 130)):
    x0, y0, x1, y1 = box
    random.seed(3)
    w = x1 - x0
    h = y1 - y0
    pts = []
    n = 40
    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * w
        y = y0 + h*0.5 + math.sin(t * 9 + 1) * h * 0.32 * (1 - abs(t-0.5))
        pts.append((x, y))
    draw.line(pts, fill=color, width=2, joint="curve")
    # small flourish underline
    draw.line([(x0+w*0.1, y1-4), (x1-w*0.1, y1-4)], fill=color, width=1)


def draw_guilloche_background(img):
    """Fine diagonal cross-hatch engraving covering the WHOLE card -- the
    single most visually distinctive feature of image-id-card-1.jpg that a
    flat background color can't reproduce (security-document paper is never
    a flat fill). Two families of thin, low-contrast diagonal lines,
    generic/procedural -- not a copy of the real card's exact engraving.
    """
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    step = 9
    tint = (150, 160, 140, 40)
    for x in range(-h, w, step):
        odraw.line([(x, 0), (x + h, h)], fill=tint, width=1)
    for x in range(0, w + h, step):
        odraw.line([(x, 0), (x - h, h)], fill=tint, width=1)
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))


def draw_security_pattern(img):
    """Generic recreation of the rainbow-tinted 'KINGDOM OF CAMBODIA'
    security-print watermark seen on real Cambodian ID cards: a fine tiled
    diagonal phrase, shifting through muted green/yellow/orange, spanning
    almost the ENTIRE card (not one boxed corner) and sitting behind the
    foreground text, same as a real security-printed background -- a
    stylized look-alike, not a reproduction of the real card's exact
    artwork/text placement.
    """
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    font = load_font("NotoSansKhmer.ttf", 9, weight=700)
    text = "KINGDOM OF CAMBODIA"
    rainbow = [(60, 150, 90), (190, 160, 40), (200, 110, 40), (170, 70, 90)]

    # small single-phrase tile, much narrower than the card, so the nested
    # paste loop below repeats it densely across the whole surface instead
    # of a few oversized, discrete-looking blobs.
    scratch = Image.new("RGBA", (10, 10))
    tb = ImageDraw.Draw(scratch).textbbox((0, 0), text, font=font)
    tile_w, tile_h = tb[2] - tb[0] + 34, 22
    base_tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    ImageDraw.Draw(base_tile).text((10, 3), text, font=font, fill=(255, 255, 255, 255))
    base_tile = base_tile.rotate(-18, expand=True, resample=Image.BICUBIC)

    strip = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    row = 0
    for ty in range(-base_tile.height, h + base_tile.height, base_tile.height):
        col = 0
        for tx in range(-base_tile.width, w + base_tile.width, base_tile.width):
            color = rainbow[(row + col) % len(rainbow)]
            tinted = Image.new("RGBA", base_tile.size, color + (0,))
            tinted.putalpha(base_tile.getchannel("A").point(lambda a: int(a * 55 / 255)))
            strip.alpha_composite(tinted, (tx, ty))
            col += 1
        row += 1
    overlay.alpha_composite(strip, (0, 0))

    # abstract translucent arcs standing in for a hologram foil ribbon,
    # tucked in the upper-right where the real card's foil strip sits
    sdraw = ImageDraw.Draw(overlay)
    cx, cy = w * 0.62, h * 0.22
    for r in (55, 42, 30, 18):
        sdraw.arc([cx - r, cy - r, cx + r, cy + r], start=195, end=345,
                  fill=(220, 240, 250, 130), width=3)
    img.paste(overlay, (0, 0), overlay)


def build_specimen():
    ensure_fonts()
    img = Image.new("RGB", (CARD_W, CARD_H), BG)
    draw_guilloche_background(img)
    draw_security_pattern(img)
    draw = ImageDraw.Draw(img)

    f_id = ImageFont.truetype(str(FONT_DIR / "CourierPrime-Bold.ttf"), 22)
    f_khmer = load_font("NotoSansKhmer.ttf", 20)
    f_khmer_bold = load_font("NotoSansKhmer.ttf", 20, weight=600)
    f_latin = load_font("NotoSansKhmer.ttf", 16, weight=600)
    f_mrz = ImageFont.truetype(str(FONT_DIR / "CourierPrime-Regular.ttf"), 24)

    # ---- Fake data -- MRZ lines are DERIVED from these same source values
    # below (via build_td1_mrz), instead of being separately hand-typed, so
    # the front fields and the MRZ can't drift out of sync with each other.
    doc_number = "998811224"
    id_suffix = "01"
    dob_yymmdd, sex_mrz = "950612", "F"
    expiry_yymmdd = "330214"
    surname_latin, given_latin = "LEE", "SOVANNA"

    data = {
        "id_number": f"{doc_number} ({id_suffix})",
        "name_khmer": "គោត្តនាមនិងនាម: លី សុវណ្ណា",
        "name_latin": f"{surname_latin} {given_latin}",
        # "កម្ពស់" (height) -- confirmed against image-id-card-1.jpg's own
        # dob/sex/height line, which spells it with coeng-ព (កម្ពស់), not
        # the "កំពស់" typo the field used previously.
        "dob_sex_height": "ថ្ងៃខែឆ្នាំកំណើត: ១២.០៦.១៩៩៥  ភេទ: ស្រី  កម្ពស់: ១៦០ ស.ម",
        "place_of_birth": "ទីកន្លែងកំណើត: ភូមិ៣ សង្កាត់ទួលទំពូង ខណ្ឌចំការមន ភ្នំពេញ",
        "address1": "អាសយដ្ឋាន: ផ្ទះ១២ ផ្លូវ៥៦៣ ភូមិ២",
        "address2": "សង្កាត់បឹងកក់២ ខណ្ឌទួលគោក ភ្នំពេញ",
        "issue_expiry": "សុពលភាព: ១៥.០២.២០២៣ ដល់ថ្ងៃ ១៤.០២.២០៣៣",
        # image-id-card-1.jpg's final field, at that photo's resolution, reads
        # as an issuing-office reference rather than "distinguishing marks" --
        # exact wording isn't fully legible, so this is a plausible stand-in
        # matching the real form's structure, not a transcription.
        "marks": "ចេញដោយ: ការិយាល័យអត្តសញ្ញាណប័ណ្ណ ភ្នំពេញ",
    }
    data["mrz1"], data["mrz2"], data["mrz3"] = build_td1_mrz(
        doc_number, dob_yymmdd, sex_mrz, expiry_yymmdd, "KHM", surname_latin, given_latin
    )

    # Rounded-corner card silhouette (real cards aren't sharp-cornered)
    mask = Image.new("L", (CARD_W, CARD_H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, CARD_W - 1, CARD_H - 1], radius=CARD_RADIUS, fill=255)

    # ID number
    draw.text((BOX["id_number"][0], BOX["id_number"][1]), data["id_number"], font=f_id, fill=INK)

    # Photo (generic placeholder avatar, not a real photo) -- clipped to frame
    draw_generic_avatar(draw, img, BOX["photo"], seed_color=(120, 140, 170))

    # Signature (synthetic squiggle)
    draw_signature(draw, BOX["signature"])

    # Text fields
    draw.text((BOX["name_khmer"][0], BOX["name_khmer"][1]), data["name_khmer"], font=f_khmer_bold, fill=INK)
    draw.text((BOX["name_latin"][0], BOX["name_latin"][1]), data["name_latin"], font=f_latin, fill=INK)
    draw.text((BOX["dob_sex_height"][0], BOX["dob_sex_height"][1]), data["dob_sex_height"], font=f_khmer, fill=INK)
    draw.text((BOX["place_of_birth"][0], BOX["place_of_birth"][1]), data["place_of_birth"], font=f_khmer, fill=INK)
    draw.text((BOX["address1"][0], BOX["address1"][1]), data["address1"], font=f_khmer, fill=INK)
    draw.text((BOX["address2"][0], BOX["address2"][1]), data["address2"], font=f_khmer, fill=INK)
    draw.text((BOX["issue_expiry"][0], BOX["issue_expiry"][1]), data["issue_expiry"], font=f_khmer, fill=INK)
    draw.text((BOX["marks"][0], BOX["marks"][1]), data["marks"], font=f_khmer, fill=INK)

    # MRZ -- CourierPrime is a true monospace font, so its natural spacing
    # already lines every character up in a grid (an earlier attempt
    # hand-computed equal advances on top of the proportional NotoSansKhmer
    # font instead; that only looked uneven). Each row is then additionally
    # stretched to fill the full card width, matching how real MRZ rows are
    # printed edge to edge rather than left-packed.
    for key in ("mrz1", "mrz2", "mrz3"):
        draw_justified_mono_line(draw, BOX[key], data[key], f_mrz, INK)

    # ---- SPECIMEN watermark (diagonal, semi-transparent) ----
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    wm_font = load_font("NotoSansKhmer.ttf", 70, weight=700)
    wm_text = "SPECIMEN"
    tb = ImageDraw.Draw(overlay).textbbox((0, 0), wm_text, font=wm_font)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    txt_layer = Image.new("RGBA", (tw+20, th+20), (0,0,0,0))
    ImageDraw.Draw(txt_layer).text((10,10), wm_text, font=wm_font, fill=(180, 20, 20, 110))
    txt_layer = txt_layer.rotate(22, expand=True)
    overlay.alpha_composite(txt_layer, ((CARD_W-txt_layer.width)//2, (CARD_H-txt_layer.height)//2))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # clip everything to the rounded-corner card silhouette
    rounded = Image.new("RGB", img.size, (255, 255, 255))
    rounded.paste(img, (0, 0), mask)
    img = rounded
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, CARD_W - 1, CARD_H - 1], radius=CARD_RADIUS, outline=(120, 120, 120), width=1)

    # small corner tag reinforcing it's fictional
    tag_font = load_font("NotoSansKhmer.ttf", 12, weight=600)
    draw.text((6, CARD_H-18), "FICTIONAL SAMPLE - NOT A REAL DOCUMENT", font=tag_font, fill=(150,20,20))

    return img


if __name__ == "__main__":
    card = build_specimen()
    # Layout/fonts are calibrated at CARD_W x CARD_H; downscale the finished
    # render to the smaller output size rather than regenerating at a
    # different resolution, so proportions stay exact.
    out_w = round(CARD_W * OUTPUT_SCALE)
    out_h = round(CARD_H * OUTPUT_SCALE)
    card = card.resize((out_w, out_h), Image.LANCZOS)
    out_dir = SCRIPT_DIR / "samples" / "specimen_id"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "specimen_id_card.png"
    card.save(out_path)
    print("Saved:", out_path)
