# Fonts

Drop `.ttf` / `.otf` files here, grouped by script:

```
fonts/
  km/   Khmer fonts   (e.g. Khmer OS, Khmer OS Battambang, Kantumruy Pro, Noto Sans Khmer)
  en/   Latin fonts for English   (e.g. Noto Sans, DejaVu Sans, Times New Roman)
  fr/   Latin fonts for French    (French uses the same Latin glyphs as English;
                                    it's fine to symlink or reuse the en/ set —
                                    kept as a separate dir in case you want
                                    period-accurate fonts for old scanned documents)
```

`text_sampler.py` / `line_renderer.py` scan these subfolders at runtime — nothing to
register manually, just add files.

## Khmer text shaping

Khmer is a complex script: vowels and subscript consonants get reordered and
combined with base consonants at render time. Pillow only does this correctly
if it was built with **Raqm** (HarfBuzz + FriBidi) text-layout support.

Check with:

```python
from PIL import features
print(features.check("raqm"))  # must print True
```

If `False`, install a Raqm-enabled Pillow (`pip install "Pillow>=10" --no-binary :all:`
on a system with `libraqm-dev`, or use a wheel that bundles it). `line_renderer.py`
raises a warning at import time if Raqm support isn't detected, since text will
render with broken glyph reordering otherwise.

Good free Khmer fonts to start with: Noto Sans Khmer, Khmer OS, Khmer OS Siemreap,
Kantumruy Pro (all on Google Fonts / Khmer Unicode community sites).
