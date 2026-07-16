# data_gen

Synthetic document generator for training the OCR recognizer (Conformer
encoder + blockwise-AR decoder) and the YOLO layout detector.

## Layout

```
data_gen/
  config.py            image sizes, font-size ranges, layout classes
  text_sampler.py       samples km/en/fr sentences from corpora/, with
                         optional code-mixing (short EN/FR span spliced
                         into a Khmer sentence)
  line_renderer.py      renders a single text-line image + transcript
  page_layout.py        layout templates: book, research_paper, national_id,
                         id_card_front, id_card_back, birth_certificate,
                         official_letter (modeled on example_images/) —
                         defines where each region goes on the page/card
  page_renderer.py      fills a layout's regions with sampled text (or, for
                         "signature"/"photo", placeholder graphics), tracks
                         per-line and per-region boxes
  degrade.py            blur / noise / jpeg / rotation augmentation,
                         rotates label boxes along with the image
  generate_lines.py     CLI -> line-recognition dataset (image + transcript)
  generate_pages.py     CLI -> page dataset with YOLO labels + dataset.yaml
  generate_pdfs.py      CLI -> ReportLab-based PDF documents (letterhead,
                         tabular notice, corporate flyer, bilingual civil
                         registry extract) with a per-set .json sidecar of
                         sampled field values; no bounding boxes (PDF output,
                         not raster), so it's for document-level variety /
                         layout ideas rather than detector/recognizer training
  generate_letter.py     CLI -> single Pillow-rendered Khmer official-letter
                         PNG (letterhead/title/paragraphs/signature), no
                         bounding boxes -- the un-tuned baseline
                         generate_letter_v2.py was calibrated from
  generate_letter_v2.py  CLI -> Khmer official-letter dataset with YOLO
                         labels + dataset.yaml (same convention as
                         generate_pages.py); every spacing constant was
                         measured against example_images/image7.jpg
  plot_layout_tags.py    CLI -> draws a dataset's YOLO boxes back onto its
                         image with a class-color legend, for a quick visual
                         sanity check (works on generate_pages.py's or
                         generate_letter_v2.py's output)
  generate_specimen_id.py CLI -> single fictional "SPECIMEN"-watermarked
                         Cambodian ID card PNG, field layout calibrated
                         against example_images/image-id-card-1.jpg; no
                         bounding boxes yet (see its module docstring)
  fonts/{km,en,fr}/      drop .ttf/.otf files here (see fonts/MANIFEST.md)
  fonts/handwritten/{km,en,fr}/
                         handwriting-style fonts (Taprom/Freehand/Fasthand
                         for Khmer, Caveat/Patrick Hand for EN/FR) — kept
                         separate from fonts/ so printed vs. handwritten
                         generation is a deliberate choice, not a random mix
  fonts/letterhead/      Noto Sans Khmer / Moul / Dancing Script -- used only
                         by generate_letter.py / generate_letter_v2.py,
                         auto-downloaded on first run; kept separate from the
                         km/en/fr layout above since these are picked by role
                         (body/heading/signature), not by language
  corpora/{km,en,fr}/    drop .txt corpora here (see corpora/MANIFEST.md)
  backgrounds/           optional paper-texture images (page_renderer falls
                          back to a plain off-white background if empty)
  samples/               example output from generate_lines.py /
                          generate_pages.py / generate_letter_v2.py, checked
                          in for quick reference
  utils/                 bbox math, font discovery + Raqm check
```

`example_images/` (top-level, alongside `data_gen/`) holds real reference
photos — a Khmer ID card (front/back), an extract of birth certificate, and
government notices/press releases — that the `id_card_front`, `id_card_back`,
`birth_certificate`, and `official_letter` templates are modeled on.

## Quick start

```bash
pip install -r data_gen/requirements.txt
# add fonts to data_gen/fonts/{km,en,fr}/ (see MANIFEST.md there — required,
# generation will raise FileNotFoundError per-language until you do)
# optionally add corpora to data_gen/corpora/{km,en,fr}/ (falls back to a
# small built-in placeholder wordlist otherwise, just to smoke-test)

cd /home/user/khmer-asr/tuna-ocr
python -m data_gen.generate_lines --out-dir dataset/lines --num-samples 5000
python -m data_gen.generate_pages --out-dir dataset/pages --num-pages 2000
```

`dataset/pages/dataset.yaml` is ready to hand to Ultralytics directly:

```bash
yolo detect train data=dataset/pages/dataset.yaml model=yolo26n.pt epochs=100
```

(swap `yolo26n.pt` for whatever YOLO checkpoint you have available — the
generator's output format doesn't depend on the model version.)

## Detection classes

`title`, `header`, `paragraph`, `line`, `field`, `signature`, `photo`,
`stamp` — configurable in `config.py` (`LAYOUT_CLASSES`). `paragraph` is the
union box of its child `line` boxes, so a detector trained on this can be
used at either granularity (block-level layout analysis, or line-level to
feed straight into the recognizer). `field` is a single label/value row
(ID-card and form fields); `signature`, `photo`, and `stamp` are non-text
placeholder regions (a scribble, a head-and-shoulders icon, and a
translucent red seal, respectively) so the detector learns to locate — not
read — those areas.

## Visual fidelity (id_card / birth_certificate / official_letter)

These three templates go beyond plain text-on-background to loosely match
the real documents in `example_images/`:

- **id_card_front/back**: dual-tone border frame, a diagonal tiled
  "KINGDOM OF CAMBODIA" watermark, and a corner crest emblem.
- **birth_certificate**: form-table gridlines behind the field rows, a
  crest emblem, and a red stamp overlapping one of the two signatures.
- **official_letter**: a centered letterhead emblem and a stamp overlapping
  the signature.

The emblem/stamp graphics (`_draw_emblem`, `_draw_stamp` in
`page_renderer.py`) are generic — concentric rings, a 5-point star, radiating
tick marks — deliberately not a reproduction of any specific government
seal or coat of arms.

## Recognizer target format

Each sampled line carries `lang_spans`: a list of `(lang, substring)` pairs.
`generate_lines.py` serializes this as inline tags, e.g.:

```
<km>ក្រសួងអប់រំ យុវជន និងកីឡា <en>UNESCO
```

Use this to build a language-tagged subword vocabulary (`<km>`, `<en>`,
`<fr>` tokens) for the blockwise-AR decoder, so it can switch scripts
mid-line without a separate language-ID step.

## External-source data + chunking

Real (non-synthetic) OCR training data pulled from external Hugging Face
datasets, chunked for Conformer-encoder input windowing, now lives in its
own top-level package: see `real_data/README.md`. It's separate from this
directory because it pulls in real third-party data rather than generating
it.

## PDF generator (generate_pdfs.py)

A separate ReportLab-based generator for five document templates (essay
report, government announcement, tabular notice, corporate flyer, bilingual
combined registry). Unlike `generate_pages.py`, it emits real PDFs (with
ReportLab's own text flow/pagination) rather than fixed-layout raster images,
so it has no bounding-box labels — use it for visual/document variety, not
detector or recognizer training data.

```bash
python -m data_gen.generate_pdfs --out-dir data_gen/samples/pdfs --num-sets 5 --seed 0
```

Requires a Khmer `.ttf`/`.otf` under `fonts/km/` (raises `FileNotFoundError`
otherwise — a plain `Helvetica` fallback would silently render Khmer as
blank/garbled text, so this generator refuses to guess). Body/paragraph text
is sampled via `TextSampler` (same corpora as the rest of `data_gen/`) rather
than `Faker`, since `Faker` has no working `km_KH` locale; `Faker` (default
`en_US`) is used only for the fields that are genuinely meant to be
Latin-script (e.g. the English half of the combined registry, sourced from
the same sampled identity as its Khmer half).

## Letter generator (generate_letter_v2.py)

A Pillow-based generator (no PDF pagination, no ReportLab) for a single
Khmer official-letter layout: centered crown title/motto, a divider rule, a
left-aligned office/department block, a centered document title, justified
body paragraphs (one rendered bold), and a right-aligned closing block
(date, signer org, a translucent circular stamp, a cursive signature
flourish). Every margin/gap constant was measured directly off
`example_images/image7.jpg` (row/column dark-pixel projection, cross-checked
against crops of each detected text band) and converted to a fraction of
page width/height, so the layout is a close proportional match rather than a
guess. Unlike `generate_letter.py` (the earlier, un-calibrated version, kept
for reference), `v2` also tracks a bounding box for every region it draws
and emits YOLO labels + `dataset.yaml`, using the same `LAYOUT_CLASSES` as
`generate_pages.py` (`title` for the crown title and doc title, `header` for
the office block, `paragraph`/`line` for body text, `signature` for the
flourish, `stamp` for the seal).

```bash
python -m data_gen.generate_letter_v2 --num-samples 20 --seed 0
python -m data_gen.plot_layout_tags --dataset-dir data_gen/samples/letterhead_dataset --name letter_00000
```

The letterhead org/department, document title, closing date, signer
org/title, and signature name are all drawn from small placeholder pools
(`ORG_NAMES`, `DEPT_NAMES`, `DOC_TITLES`, `SIGNER_ORGS`, `SIGNER_NAMES` in
`generate_letter_v2.py`) and re-randomized per sample via `--seed`, so a
detector trained on this data doesn't just memorize one fixed string per
region. The crown title itself (`ព្រះរាជាណាចក្រកម្ពុជា` / `ជាតិ សាសនា
ព្រះមហាក្សត្រ`) is left fixed -- it's the national letterhead motto, not
something that varies letter-to-letter. The stamp (`draw_stamp`) is the same
generic rings-and-radiating-ticks design as `page_renderer.py`'s
`_draw_stamp` -- not a reproduction of any real organization's seal or text.

## Known gaps / next steps

- Khmer word-wrapping inside a region is not implemented — each "line" is
  one sampled sentence shrunk to fit the region width, not a paragraph
  re-flowed word-by-word. Fine for training a layout/line detector; if you
  need faithful multi-line paragraph wrapping, extend `page_renderer.py`'s
  paragraph loop to split a longer sampled paragraph into wrapped segments.
- No table/figure/footer/page-number region types yet — add templates in
  `page_layout.py` and classes in `config.LAYOUT_CLASSES` as needed.
- `backgrounds/` is empty by default; add scanned-paper textures for more
  realistic national-ID/book backgrounds.
