"""Font discovery + Raqm (complex-script shaping) sanity check."""
import random
import warnings
from pathlib import Path

from PIL import features

from ..config import FONTS_DIR

_RAQM_WARNED = False


def check_raqm():
    global _RAQM_WARNED
    if not features.check("raqm") and not _RAQM_WARNED:
        warnings.warn(
            "Pillow was built without Raqm (HarfBuzz+FriBidi) support. "
            "Khmer text will render with incorrect vowel/subscript reordering. "
            "See data_gen/fonts/MANIFEST.md for how to fix this.",
            stacklevel=2,
        )
        _RAQM_WARNED = True


class FontBank:
    """Scans fonts/<lang>/*.{ttf,otf} and hands out random choices per language."""

    def __init__(self, fonts_dir: Path = FONTS_DIR):
        check_raqm()
        self.fonts = {}
        for lang in ("km", "en", "fr"):
            lang_dir = fonts_dir / lang
            paths = []
            if lang_dir.is_dir():
                paths = sorted(lang_dir.glob("*.ttf")) + sorted(lang_dir.glob("*.otf"))
            self.fonts[lang] = paths

    def has_fonts(self, lang: str) -> bool:
        return len(self.fonts.get(lang, [])) > 0

    def random_font_path(self, lang: str) -> Path:
        candidates = self.fonts.get(lang) or []
        if not candidates:
            raise FileNotFoundError(
                f"No fonts found for language '{lang}' in {FONTS_DIR / lang}. "
                "Add .ttf/.otf files there (see fonts/MANIFEST.md)."
            )
        return random.choice(candidates)
