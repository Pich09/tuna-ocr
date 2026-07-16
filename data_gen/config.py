"""Central config for the synthetic document generator."""
from dataclasses import dataclass, field
from pathlib import Path

DATA_GEN_ROOT = Path(__file__).resolve().parent
FONTS_DIR = DATA_GEN_ROOT / "fonts"
CORPORA_DIR = DATA_GEN_ROOT / "corpora"
BACKGROUNDS_DIR = DATA_GEN_ROOT / "backgrounds"

# Region classes for the layout detector (YOLO class indices follow list order).
LAYOUT_CLASSES = ["title", "header", "paragraph", "line", "field", "signature", "photo", "stamp"]

# Decorative-graphic settings for id_card/birth_certificate/official_letter
# templates, tuned to loosely resemble example_images/ (a generic circular
# crest, a diagonal repeated watermark, and a red official-stamp circle —
# not a reproduction of any specific government seal).
WATERMARK_TEXT = "KINGDOM OF CAMBODIA"
ID_CARD_BORDER_COLORS = ((30, 60, 120), (185, 150, 45))  # blue, gold
STAMP_COLOR = (185, 30, 30)

LANGUAGES = ("km", "en", "fr")

# Probability that a Khmer line gets a short EN/FR span spliced in
# (e.g. a proper noun, an acronym, a citation) to mimic real mixed-script text.
CODE_MIX_PROB = 0.15


@dataclass
class LineRenderConfig:
    img_height: int = 64
    min_font_size: int = 28
    max_font_size: int = 44
    padding_x: int = 8
    padding_y: int = 6
    text_color_range: tuple = ((0, 0, 0), (60, 60, 60))
    bg_color_range: tuple = ((235, 235, 235), (255, 255, 255))


@dataclass
class PageRenderConfig:
    page_size: tuple = (1240, 1754)  # ~A4 at 150dpi
    id_card_size: tuple = (1013, 638)  # ~ID-1 card ratio at ~300dpi
    margin: int = 90
    line_height: int = 40
    header_font_size: tuple = (34, 46)
    title_font_size: tuple = (44, 60)
    paragraph_font_size: tuple = (24, 30)
    field_font_size: tuple = (18, 24)
    line_spacing: float = 1.35
    paragraph_gap: int = 24
    classes: list = field(default_factory=lambda: list(LAYOUT_CLASSES))


# Common Khmer form/ID-card field labels, used by the id_card and
# birth_certificate templates to build realistic "label: value" rows.
FIELD_LABELS_KM = [
    "ឈ្មោះ",
    "ថ្ងៃខែឆ្នាំកំណើត",
    "ភេទ",
    "សញ្ជាតិ",
    "អាសយដ្ឋាន",
    "លេខអត្តសញ្ញាណប័ណ្ណ",
    "ថ្ងៃផុតកំណត់",
    "ជាតិ សាសនា",
    "ទីកន្លែងកំណើត",
    "ឈ្មោះឪពុក",
    "ឈ្មោះម្តាយ",
]
