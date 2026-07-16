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
  requirements.txt
  samples/                       example output from generate_external_chunks.py
```

## Quick start

```bash
pip install -r real_data/requirements.txt   # datasets, huggingface_hub, Pillow, tqdm
python -m real_data.generate_external_chunks --source darayut_scene_text --num-samples 100
python -m real_data.generate_external_chunks --source all --num-samples 50
```

## External-source data + chunking

Pulls a small streamed sample from five external Hugging Face OCR datasets
(`config.EXTERNAL_DATASETS`: `deepcopy/khmer-text-recognition`,
`Chanrith123333/khmer_english_ocr_image_line`,
`Darayut/khmer-scene-text-synthetic-contrast`,
`SoyVitou/khmer-handwritten-dataset-4.2k`, `Sokheng/khmer-synthetic-ocr-v1-100k`)
and slices each line image into fixed-width overlapping chunks
(`chunking.py`) for Conformer-encoder input windowing.

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

All 5 sources' exact column names are undocumented (their HF dataset
viewers are broken or missing), so their `image_col`/`text_col` are `None`
in `EXTERNAL_DATASETS` and get auto-detected at runtime (first `PIL.Image`
column, first remaining `str` column); ambiguous detection raises rather
than guessing. A source that fails to load (network, gated, schema
mismatch) is skipped with a clear message under `--source all`, or
hard-fails under a single named `--source`.
