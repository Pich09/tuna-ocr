# real_data

Real (non-synthetic) OCR training data pipeline: pulls samples from
external Hugging Face datasets and slices each line image into fixed-width
overlapping chunks for Conformer-encoder input windowing. Kept as its own
top-level package, separate from `data_gen/` (the synthetic document
generator), since this pulls in real third-party data rather than
generating it.

```
real_data/
  config.py                     REAL_DATA_ROOT, ExternalChunkConfig, EXTERNAL_DATASETS
  chunking.py                   chunk_image_overlap() -- pure PIL.Image -> chunks, no network deps
  hf_datasets.py                 HF `datasets` streaming loader + column auto-detect
  generate_external_chunks.py   CLI -> images + manifest.tsv
  dedup.py                       exact (SHA-256) + near-duplicate (perceptual hash) detection -- pure functions
  deduplicate.py                 CLI -> pools + deduplicates one or more pulled sources into one manifest
  requirements.txt
  samples/                       example output from generate_external_chunks.py / deduplicate.py
```

## Quick start

```bash
pip install -r real_data/requirements.txt   # datasets, huggingface_hub, Pillow, tqdm, ImageHash
python -m real_data.generate_external_chunks --source darayut_scene_text --num-samples 100
python -m real_data.generate_external_chunks --source all --num-samples 50

# then deduplicate before handing data to recognizer.train (see below) --
# list each pulled source explicitly, not a glob (a glob would also match
# the dedup/ output directory itself once created)
python -m real_data.deduplicate \
    --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \
                     real_data/samples/chanrith_ocr_image_line \
                     real_data/samples/darayut_scene_text \
                     real_data/samples/sokheng_synthetic_v1 \
    --out-dir real_data/samples/dedup
```

## External-source data + chunking

Pulls a small streamed sample from four external Hugging Face OCR datasets
(`config.EXTERNAL_DATASETS`: `deepcopy/khmer-text-recognition`,
`Chanrith123333/khmer_english_ocr_image_line`,
`Darayut/khmer-scene-text-synthetic-contrast`, `Sokheng/khmer-synthetic-ocr-v1-100k`)
and slices each line image into fixed-width overlapping chunks
(`chunking.py`) for Conformer-encoder input windowing. All four are printed
text -- `SoyVitou/khmer-handwritten-dataset-4.2k` (the only handwritten
source) was deliberately removed from the registry so training focuses on
printed text first; add a dedicated handwritten pass back in later rather
than blending a single handwritten source in with these.

None of these datasets provide per-character/word boxes -- only one
whole-line transcript per image -- so chunking does **not** re-derive
per-chunk labels. Each chunk of a line shares that line's single whole-line
transcript (`manifest.tsv`'s `text` column repeats per chunk, grouped by
`line_id`); chunking only bounds each block's pixel width, matching how a
blockwise-AR decoder (see `data_gen/README.md`'s "Recognizer target format")
consumes bounded-width encoder blocks. Chunk i (i>=1) starts `overlap` px
before where a naive non-overlapping slice would put it, so a glyph
straddling the previous chunk's right edge is captured whole in the next
chunk too. `sokheng_synthetic_v1` alone is ~100k rows, so loading always
streams rather than downloading the full dataset -- `--num-samples` caps
how many lines are pulled, not a subset selection. By default samples come out in the
dataset's natural streamed order (`--shuffle-buffer 0`); these datasets
embed images directly in parquet rows, and a single row can take tens of
seconds to fetch over a slow connection, so a streaming shuffle buffer --
which must fully fill, one fetched row at a time, before anything is
yielded -- can turn into minutes of wait even for a small buffer. Pass
`--shuffle-buffer N` (N>0) if you have a fast connection and want less
order-correlated sampling.

All 4 sources' exact column names are undocumented (their HF dataset
viewers are broken or missing), so their `image_col`/`text_col` are `None`
in `EXTERNAL_DATASETS` and get auto-detected at runtime (first `PIL.Image`
column, first remaining `str` column); ambiguous detection raises rather
than guessing. A source that fails to load (network, gated, schema
mismatch) is skipped with a clear message under `--source all`, or
hard-fails under a single named `--source`.

## Deduplication (`dedup.py` / `deduplicate.py`)

Run `real_data.deduplicate` after pulling and before handing data to
`recognizer.train` -- it pools the `is_full_line` (whole-line) rows across
however many pulled source directories you give it and removes:

- **Exact duplicates** (SHA-256 of the raw image bytes) -- e.g. the same
  row appearing twice, or two sources that turn out to mirror the same
  underlying data under different names/accounts.
- **Near-duplicates** (`imagehash.phash`, Hamming distance <=
  `--near-dup-threshold`, default 4) -- e.g. a re-compressed/re-encoded
  copy that isn't byte-identical but is the same line image.

It does **not** remove same-transcript-different-image rows -- many
distinct images can legitimately share a transcript (common Khmer words
rendered by different writers/fonts/backgrounds), so that's reported as an
informational count (`duplicate_text_count_informational` in the JSON
report), not filtered.

Output: a single pooled `manifest.tsv` (columns: `image_path` [absolute],
`text`, `source`, `line_id`) plus `dedup_report.json` (counts, per-source
survivor counts, and up to 20 example duplicate groups of each kind, for
spot-checking). `recognizer.train`'s `--dedup-manifest` flag (preferred
over `--real-data-dirs`) points straight at this manifest.

Near-duplicate detection is brute-force pairwise comparison over the
exact-dedup survivors -- O(n²), fine at the scale these pulls operate at
(hundreds to a few thousand samples per source) but not something that
would scale to a full multi-million-row pull without an LSH/BK-tree index;
noted as a known limitation in `dedup.py`'s docstring, not implemented
since it isn't needed at current pull sizes.
