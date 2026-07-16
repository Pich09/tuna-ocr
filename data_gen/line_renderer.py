"""Renders a single text-line image (+ transcript) for OCR-recognizer training."""
import random

from PIL import Image, ImageDraw, ImageFont

from .config import LineRenderConfig
from .utils.fonts import FontBank


def _random_color(lo, hi):
    return tuple(random.randint(a, b) for a, b in zip(lo, hi))


def _load_font(font_path, size):
    try:
        return ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    except AttributeError:
        # Older Pillow without ImageFont.Layout enum.
        return ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.LAYOUT_RAQM)


def render_line(text: str, lang: str, font_bank: FontBank, cfg: LineRenderConfig = LineRenderConfig()):
    """Returns (PIL.Image, dict metadata). Raises FileNotFoundError if no font
    is registered for `lang` — see fonts/MANIFEST.md.
    """
    font_size = random.randint(cfg.min_font_size, cfg.max_font_size)
    font_path = font_bank.random_font_path(lang)
    font = _load_font(font_path, font_size)

    text_color = _random_color(*cfg.text_color_range)
    bg_color = _random_color(*cfg.bg_color_range)

    # Measure first on a throwaway canvas, then render at exact size.
    scratch = Image.new("RGB", (10, 10), bg_color)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w = text_w + 2 * cfg.padding_x
    img_h = max(cfg.img_height, text_h + 2 * cfg.padding_y)

    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((cfg.padding_x - bbox[0], cfg.padding_y - bbox[1]), text, font=font, fill=text_color)

    meta = {
        "text": text,
        "lang": lang,
        "font": font_path.name,
        "font_size": font_size,
        "width": img_w,
        "height": img_h,
    }
    return img, meta
