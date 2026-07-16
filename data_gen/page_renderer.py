"""Composes a full synthetic page (book / research paper / national ID /
ID card / birth certificate / official letter) from a page_layout template,
filling each region with sampled text, and returns the page image plus
per-region YOLO-format detection boxes (classes: title, header, paragraph,
line, field, signature, photo) and per-line transcripts (for reuse as extra
recognizer training data, cropped straight out of the page).
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .config import (
    PageRenderConfig,
    LAYOUT_CLASSES,
    BACKGROUNDS_DIR,
    FIELD_LABELS_KM,
    WATERMARK_TEXT,
    ID_CARD_BORDER_COLORS,
    STAMP_COLOR,
)
from .line_renderer import _load_font, _random_color
from .page_layout import sample_layout
from .text_sampler import TextSampler
from .utils.bbox import xyxy_to_yolo, union
from .utils.fonts import FontBank

CARD_TEMPLATES = ("id_card_front", "id_card_back")
STAMPED_TEMPLATES = ("birth_certificate", "official_letter")


def _fit_font(draw, text, font_path, box_w, max_size, min_size=12):
    """Shrinks font size until the text fits within box_w."""
    size = max_size
    while size > min_size:
        font = _load_font(font_path, size)
        w = draw.textbbox((0, 0), text, font=font)[2]
        if w <= box_w:
            return font, size
        size -= 2
    return _load_font(font_path, min_size), min_size


def _background(cfg: PageRenderConfig, size):
    bg_files = sorted(BACKGROUNDS_DIR.glob("*.png")) + sorted(BACKGROUNDS_DIR.glob("*.jpg"))
    if bg_files:
        img = Image.open(random.choice(bg_files)).convert("RGB").resize(size)
        return img
    return Image.new("RGB", size, (250, 249, 245))


def _sample_field_text(text_sampler: TextSampler):
    label = random.choice(FIELD_LABELS_KM)
    value = " ".join(text_sampler.sample_sentence("km").split()[:3])
    return f"{label} ⁖ {value}"


def _draw_signature(draw, box, color=(20, 20, 60)):
    """Draws a squiggly scribble to stand in for a handwritten signature."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    n_points = random.randint(8, 14)
    points = [
        (x0 + random.uniform(0, w), y0 + h * 0.3 + random.uniform(-h * 0.35, h * 0.35))
        for _ in range(n_points)
    ]
    points.sort(key=lambda p: p[0])
    draw.line(points, fill=color, width=2, joint="curve")


def _draw_photo_placeholder(draw, box, border=(120, 120, 120), fill=(225, 225, 225)):
    """Draws a plain photo-slot box with a simple head-and-shoulders glyph."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=fill, outline=border, width=2)
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2, y0 + h * 0.35
    r = min(w, h) * 0.18
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(190, 190, 190))
    shoulder_top = cy + r * 1.3
    draw.pieslice(
        (cx - w * 0.32, shoulder_top, cx + w * 0.32, shoulder_top + h * 0.5),
        180, 360, fill=(190, 190, 190),
    )


def _draw_card_border(draw, size, colors):
    """Dual-tone decorative frame, loosely matching the trim on a real ID card."""
    w, h = size
    draw.rectangle((4, 4, w - 5, h - 5), outline=colors[0], width=4)
    draw.rectangle((10, 10, w - 11, h - 11), outline=colors[1], width=2)


def _draw_emblem(draw, center, r, primary=ID_CARD_BORDER_COLORS[1], secondary=ID_CARD_BORDER_COLORS[0]):
    """Generic circular crest placeholder (concentric rings + a 5-point
    star) — a stand-in for a ministry/state emblem, not a reproduction of
    any specific coat of arms.
    """
    cx, cy = center
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=primary, width=3)
    draw.ellipse((cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7), outline=secondary, width=2)
    pts = []
    for i in range(5):
        ang = -math.pi / 2 + i * 2 * math.pi / 5
        pts.append((cx + r * 0.45 * math.cos(ang), cy + r * 0.45 * math.sin(ang)))
        ang2 = ang + math.pi / 5
        pts.append((cx + r * 0.18 * math.cos(ang2), cy + r * 0.18 * math.sin(ang2)))
    draw.polygon(pts, fill=primary)


def _draw_watermark(img, box, text, font_path, angle=-25):
    """Tiles `text`, rotated, at low opacity across `box` — mimics the
    diagonal security-pattern watermark seen on real ID cards.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return

    font = _load_font(font_path, 22)
    scratch = Image.new("RGBA", (10, 10))
    tb = ImageDraw.Draw(scratch).textbbox((0, 0), text, font=font)
    tile = Image.new("RGBA", (tb[2] - tb[0] + 60, tb[3] - tb[1] + 60), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((30 - tb[0], 30 - tb[1]), text, font=font, fill=(110, 110, 110, 45))
    tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for ty in range(-tile.height, h + tile.height, tile.height):
        for tx in range(-tile.width, w + tile.width, tile.width):
            overlay.alpha_composite(tile, (tx, ty))
    img.paste(overlay, (x0, y0), overlay)


def _draw_stamp(img, box, color=STAMP_COLOR):
    """Draws a translucent red circular seal (rings + radiating ticks),
    the way a real ink stamp typically overlaps a signature.
    """
    x0, y0, x1, y1 = box
    size = min(x1 - x0, y1 - y0)
    if size <= 0:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = size / 2
    alpha = 130

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color + (alpha,), width=4)
    d.ellipse((cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75), outline=color + (alpha,), width=2)
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


def _draw_table_grid(draw, box, n_rows, col_split=0.35, color=(130, 130, 130)):
    """Draws form-table gridlines behind a column of field rows."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=color, width=2)
    row_h = (y1 - y0) / n_rows
    for i in range(1, n_rows):
        y = y0 + i * row_h
        draw.line((x0, y, x1, y), fill=color, width=1)
    col_x = x0 + (x1 - x0) * col_split
    draw.line((col_x, y0, col_x, y1), fill=color, width=1)


def render_page(
    font_bank: FontBank,
    text_sampler: TextSampler,
    cfg: PageRenderConfig = PageRenderConfig(),
    template: str = None,
    primary_lang: str = "km",
):
    template_name, regions, canvas_size = sample_layout(cfg, template)
    img = _background(cfg, canvas_size)
    draw = ImageDraw.Draw(img)
    text_color = _random_color((0, 0, 0), (50, 50, 50))

    detections = []  # (class_name, (x0, y0, x1, y1))
    line_records = []  # per-line transcript + box, for recognizer training

    if template_name in CARD_TEMPLATES:
        _draw_card_border(draw, canvas_size, ID_CARD_BORDER_COLORS)
        try:
            wm_font = font_bank.random_font_path("en")
        except FileNotFoundError:
            wm_font = font_bank.random_font_path(primary_lang)
        _draw_watermark(img, (0, 0, canvas_size[0], canvas_size[1]), WATERMARK_TEXT, wm_font)

    if template_name == "id_card_front":
        _draw_emblem(draw, (canvas_size[0] - 55, 55), 32)
    elif template_name == "birth_certificate":
        _draw_emblem(draw, (75, 65), 38)
    elif template_name == "official_letter":
        _draw_emblem(draw, (canvas_size[0] // 2, 58), 40)

    if template_name == "birth_certificate":
        field_boxes = [r["box"] for r in regions if r["type"] == "field"]
        if field_boxes:
            _draw_table_grid(draw, union(field_boxes), len(field_boxes))

    for region in regions:
        rtype = region["type"]
        x0, y0, x1, y1 = region["box"]
        box_w = x1 - x0

        if rtype in ("title", "header"):
            font_range = cfg.title_font_size if rtype == "title" else cfg.header_font_size
            text, spans = text_sampler.sample_line(primary_lang)
            font_path = font_bank.random_font_path(primary_lang)
            font, size = _fit_font(draw, text, font_path, box_w, font_range[1], font_range[0] - 6)
            bbox = draw.textbbox((x0, y0), text, font=font)
            draw.text((x0, y0), text, font=font, fill=text_color)
            detections.append((rtype, bbox))

        elif rtype == "paragraph":
            n_lines = region.get("n_lines", 1)
            line_boxes = []
            cursor_y = y0
            line_h = cfg.paragraph_font_size[1] * cfg.line_spacing
            for _ in range(n_lines):
                if cursor_y + line_h > y1:
                    break
                text, spans = text_sampler.sample_line(primary_lang)
                font_path = font_bank.random_font_path(primary_lang)
                font, size = _fit_font(
                    draw, text, font_path, box_w, cfg.paragraph_font_size[1], min_size=10
                )
                bbox = draw.textbbox((x0, cursor_y), text, font=font)
                draw.text((x0, cursor_y), text, font=font, fill=text_color)
                detections.append(("line", bbox))
                line_boxes.append(bbox)
                line_records.append({"text": text, "lang_spans": spans, "box": bbox})
                cursor_y += line_h
            if line_boxes:
                detections.append(("paragraph", union(line_boxes)))

        elif rtype == "field":
            text = _sample_field_text(text_sampler)
            font_path = font_bank.random_font_path(primary_lang)
            font, size = _fit_font(draw, text, font_path, box_w, cfg.field_font_size[1], min_size=10)
            bbox = draw.textbbox((x0, y0), text, font=font)
            draw.text((x0, y0), text, font=font, fill=text_color)
            detections.append(("field", bbox))
            line_records.append({"text": text, "lang_spans": [(primary_lang, text)], "box": bbox})

        elif rtype == "signature":
            _draw_signature(draw, (x0, y0, x1, y1), color=text_color)
            detections.append(("signature", (x0, y0, x1, y1)))

        elif rtype == "photo":
            _draw_photo_placeholder(draw, (x0, y0, x1, y1))
            detections.append(("photo", (x0, y0, x1, y1)))

    if template_name in STAMPED_TEMPLATES:
        sig_boxes = [box for cls, box in detections if cls == "signature"]
        if sig_boxes:
            stamp_box = _draw_stamp(img, random.choice(sig_boxes))
            if stamp_box:
                detections.append(("stamp", stamp_box))

    return {
        "image": img,
        "template": template_name,
        "detections": detections,
        "lines": line_records,
    }


def write_sample(out_dir: Path, name: str, sample: dict, cfg: PageRenderConfig = PageRenderConfig()):
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img = sample["image"]
    img_w, img_h = img.size
    img.save(images_dir / f"{name}.jpg", quality=92)

    label_lines = []
    for cls_name, box in sample["detections"]:
        cls_id = LAYOUT_CLASSES.index(cls_name)
        xc, yc, w, h = xyxy_to_yolo(box, img_w, img_h)
        label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    (labels_dir / f"{name}.txt").write_text("\n".join(label_lines), encoding="utf-8")
