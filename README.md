# tuna-ocr

A Khmer OCR system: synthetic + real training-data generation, and a
from-scratch PyTorch recognizer model (Conformer encoder + blockwise-AR
decoder) trained on real Hugging Face OCR datasets.

## Contents

- [Packages](#packages)
- [Model architecture](#model-architecture)
- [Quick start](#quick-start)
- [Training on GPU / TPU / CPU](#training-on-gpu--tpu--cpu)
- [Tokenizer](#tokenizer)
- [Known gaps](#known-gaps)
- [License](#license)

## Packages

```
tuna-ocr/
  data_gen/         synthetic document generator (lines, pages, ID cards, letters,
                     birth certificates) for the YOLO layout detector and OCR
                     line recognizer -- see data_gen/README.md
  real_data/        pulls line images + transcripts from 5 external Hugging Face
                     OCR datasets, slices them into overlapping chunks for
                     Conformer-encoder input windowing -- see real_data/README.md
  recognizer/       the OCR recognizer model itself: Conformer encoder +
                     blockwise-AR decoder, trained on real_data's output, using
                     the shared Panhapich/khmer-sp-8k tokenizer -- see
                     recognizer/README.md
  notebooks/        Colab/Kaggle/local training notebook + a data-pipeline
                     exploration notebook
  example_images/   real reference photos (ID cards, birth certificate,
                     official letters) the data_gen templates are modeled on
```

`recognizer/` trains **only** on `real_data`'s 5 configured Hugging Face
sources; `data_gen`'s synthetic output is a separate, independent
data-generation track (for a YOLO layout detector / future recognizer
training), not currently wired into `recognizer/`'s training pipeline.

## Model architecture

```
line image (H=64px, variable width)
        |
        v
  chunk_image_overlap()               <- fixed-width overlapping chunks
  (chunk_width=128px, overlap=16px)      are the encoder's literal input
        |                                unit, not optional augmentation
        v
  +-------------------------+
  |   Conv2dSubsampling     |  two stride-2 Conv2d layers: (1,64,128) -> (D,16,32)
  |   (per chunk, batched)  |  height collapsed into feature dim -> (32 frames, d_model)
  +-------------------------+
        |
        v
  +-------------------------+
  |  ConformerEncoder x8     |  Macaron-style block per layer:
  |  (per chunk)             |    FF/2 -> self-attn -> depthwise-conv(k=15)+GLU -> FF/2
  +-------------------------+
        |
        v
  trim + stitch chunks        <- each non-first chunk's leading ~4 frames
  (chunk index == block index)   (from its overlap left-context) are dropped
        |                        before concatenation, so no duplicated content
        v
  enc_out: (T', d_model) ------------------------+
        |                                        |
        v                                        v
  +---------------+                    +---------------------------+
  |   CTC head     |  auxiliary loss   |   BlockwiseARDecoder x4     |
  |  Linear(D,V+1) |  (monotonic       |  causal self-attn +         |
  +---------------+   alignment)       |  block-restricted cross-attn|
                                       +---------------------------+
                                                    |
                                                    v
                                       one block of up to K=8 tokens
                                       per decoding step (parallel
                                       within block, autoregressive
                                       across blocks -- see below)
```

**Chunking is the model's input contract, not optional augmentation.**
Every line image is sliced into fixed-width, overlapping chunks
(`chunk_width=128px`, `overlap=16px`, `ModelConfig`) before it reaches the
encoder. Chunk boundaries double 1:1 as the decoder's block boundaries.
Chunking is reused directly from `real_data.chunking.chunk_image_overlap`
(already source-agnostic) rather than duplicated, but with its own 128/16
setting, independent of `real_data`'s own standalone-pipeline default
(256px/32px).

**Encoder**: a `Conv2dSubsampling` front-end (two stride-2 `Conv2d` layers)
turns each fixed-width chunk into a short frame sequence — height is
collapsed into the feature dimension (a single text line has no
independent second time axis), width is reduced ~4x. Chunks are encoded
densely as one flattened batch, then an 8-layer Macaron-style
`ConformerBlock` stack (feed-forward → self-attention → depthwise
conv+GLU → feed-forward) processes each chunk. Per-chunk outputs are then
stitched back into one per-line sequence: each non-first chunk's leading
frames (from its backward-overlap left-context pixels) are trimmed before
concatenation, so the stitched sequence has no duplicated content.
Absolute sinusoidal positional encoding is used (not the full
Conformer-paper relative-position attention — a deliberate v1
simplification).

**Blockwise-AR decoder**: predicts a whole block of tokens per decoding
step (Blockwise Parallel Decoding, Stern et al. 2018, adapted here)
instead of one token at a time. Autoregression happens *between* blocks —
each block's `max_tokens_per_block=8` token slots are all produced from a
single shared hidden state via parallel output heads, so a slot never
depends on another slot's true identity from the same block (no leakage,
no inference-time chicken-and-egg problem). Cross-attention for block `b`
is restricted to encoder frames from blocks `0..b` (never future blocks).
Since no dataset provides real per-token alignment, each sample's token
sequence is uniform-split into one run per chunk/block for teacher
forcing — a documented v1 approximation.

**Loss**: hybrid CTC (auxiliary, over the full non-windowed encoder output,
weight 0.3) + AR cross-entropy. The CTC term induces the left-to-right
monotonic image/text correspondence the block-restricted cross-attention
depends on, and doubles as an independent greedy-decode sanity check.

**Defaults** (`recognizer/config.py`'s `ModelConfig`): `d_model=256`,
8 encoder layers (4 attention heads, conv kernel 15), 4 decoder layers
(4 attention heads), `max_tokens_per_block=8`. See `recognizer/README.md`
for the full per-module breakdown and known gaps.

## Quick start

```bash
pip install -r data_gen/requirements.txt -r real_data/requirements.txt -r recognizer/requirements.txt

# pull real training data (5 configured sources, see real_data/config.py)
python -m real_data.generate_external_chunks --source all --num-samples 500

# fetch the shared tokenizer
python -m recognizer.tokenizer.fetch_tokenizer --out-dir recognizer/tokenizer/assets

# train
python -m recognizer.train \
    --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \
                     real_data/samples/chanrith_ocr_image_line \
                     real_data/samples/darayut_scene_text \
                     real_data/samples/soyvitou_handwritten \
                     real_data/samples/sokheng_synthetic_v1 \
    --tokenizer-dir recognizer/tokenizer/assets --run-name v1

# evaluate / run inference
python -m recognizer.evaluate --checkpoint recognizer/checkpoints/v1/last.pt \
    --real-data-dirs real_data/samples/darayut_scene_text --tokenizer-dir recognizer/tokenizer/assets
python -m recognizer.infer --checkpoint recognizer/checkpoints/v1/last.pt \
    --image path/to/line.jpg --tokenizer-dir recognizer/tokenizer/assets
```

Or use `notebooks/train_recognizer.ipynb`, which works unmodified on Colab,
Kaggle, and locally — see its first cells for per-platform one-time setup
(HF token secret, GPU/TPU runtime selection, Kaggle internet access).

## Training on GPU / TPU / CPU

The training device is auto-detected (`recognizer/env_utils.py`) — no flag
needed on any platform:
- **GPU**: batch size is auto-probed to fit available VRAM
  (`--auto-batch-size` / the notebook's `auto_batch_size=True`).
- **TPU**: select the TPU runtime/accelerator on Colab or Kaggle *before*
  starting (both ship `torch_xla` preinstalled there). Runs on a single
  TPU core for now — see `recognizer/README.md`'s known gaps for
  multi-core support status and a CTC-op compatibility caveat.
- **CPU**: works, just slow — fine for the kind of small smoke-test run
  used to validate the pipeline end-to-end.

Checkpoints are pushed to the `Panhapich/tuna-ocr` Hugging Face repo every
10,000 steps (created **private** by default).

## Tokenizer

Uses the project's shared tokenizer, `Panhapich/khmer-sp-8k` (SentencePiece
unigram, 8000 vocab), via its upstream `KhmerTokenizer` wrapper (word
segmentation + gazetteer/loanword masking) — never a bare
`sentencepiece.SentencePieceProcessor`, or tokenization is silently wrong.
`<km>`/`<en>`/`<fr>` language tags and `<eob>` (end-of-block) are appended
past the base vocab; see `recognizer/README.md` for the full encode/decode
API.

## Known gaps

- No word-level bounding boxes for `data_gen`'s ID-card generator yet.
- Relative-position MHSA, beam search decoding, CTC-based re-alignment of
  block boundaries, KV-cached blockwise decoding, and multi-core TPU
  parallelism are all deferred — see `recognizer/README.md`'s "Known gaps
  / next steps" for the full list and rationale.

## License

MIT — see `LICENSE`.
