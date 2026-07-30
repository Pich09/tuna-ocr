# recognizer

The OCR recognizer model itself: a Conformer encoder + blockwise-AR decoder,
plain PyTorch, trained **only** on the 5 real (Hugging Face) data sources
configured in `real_data/config.py` -- `data_gen`'s synthetic images are not
part of this training pipeline.

## Layout

```
recognizer/
  config.py                 ModelConfig, TrainConfig
  env_utils.py               Colab/Kaggle/local environment detection, Drive mount, HF secrets
  hf_push.py                  pushes a checkpoint to the Panhapich/tuna-ocr HF repo
  tokenizer/
    fetch_tokenizer.py        CLI: downloads Panhapich/khmer-sp-8k assets
    khmer_ocr_tokenizer.py     KhmerOcrTokenizer wrapper
    assets/                    downloaded artifacts (gitignored)
  data/
    manifest.py                Sample dataclass + real_data manifest loader
    transforms.py               image load/normalize + chunking (chunk_image_overlap reused from real_data)
    dataset.py                  OCRLineDataset, collate_fn, BucketBatchSampler
  modules/
    attention.py                masked multi-head attention (supports per-batch masks)
    positional.py                sinusoidal positional encoding
    conv_subsampling.py          Conv2d frontend, image -> frame sequence
    conformer_block.py           Macaron-style Conformer block
    encoder.py                   ConformerEncoder (per-chunk pass + stitching)
    decoder.py                   BlockwiseARDecoder
    model.py                     Recognizer (encoder + CTC head + decoder)
  train.py                    CLI + run_training() callable
  evaluate.py                  CLI: batch CER evaluation
  infer.py                     CLI: single-image inference
  checkpoints/                 gitignored
```

## Chunking is the model's input contract

Every line image is sliced into fixed-width, overlapping chunks
(`chunk_width=128px`, `overlap=16px`, `ModelConfig`) before it reaches the
encoder -- this is not optional augmentation. Chunk boundaries double 1:1 as
the blockwise-AR decoder's block boundaries. Chunking reuses
`real_data.chunking.chunk_image_overlap` directly (it's already
source-agnostic) rather than duplicating the logic, but with its own
128/16 setting, independent of `real_data/config.py`'s `ExternalChunkConfig`
(256/32, tuned for `real_data`'s own standalone CLI).

## Blockwise-AR decoding

The decoder predicts a whole block of tokens per decoding step (Blockwise
Parallel Decoding, Stern et al. 2018, adapted here) instead of one token at
a time: autoregression happens *between* blocks, while a block's
`max_tokens_per_block` token slots are all produced from a single shared
hidden state via parallel output heads -- see `modules/decoder.py`'s
docstring for the exact mechanism and why it's leak-free. Since no dataset
provides real per-token alignment, each sample's tagged token sequence is
uniform-split into one run per chunk/block for teacher forcing -- a known
v1 approximation (see Known gaps below).

Training blockwise from step 0 teaches the decoder to reproduce that
approximate split rather than real image-to-text correspondence (observed
in practice: AR val CER stuck well above CTC's on the same encoder,
despite near-zero AR train loss). `TrainConfig.sequential_ar_steps`
(`--sequential-ar-steps` on the CLI) trains an initial fraction of steps in
a plain sequential-AR mode instead -- full teacher forcing over the true
token sequence, unrestricted cross-attention, so alignment is discovered
by attention rather than assumed from a uniform split -- then switches to
blockwise for the rest of the run. Only the output head differs per mode
(`token_head` vs `block_head`); the shared trunk (embeddings, positional
encoding, decoder layers) carries over. See `modules/decoder.py`'s
"Two-stage training" docstring section for the full mechanism.

## Tokenizer

Uses the project's shared tokenizer, `Panhapich/khmer-sp-8k`
(SentencePiece unigram, 8000 vocab) via its upstream `KhmerTokenizer`
wrapper (word segmentation + gazetteer/loanword masking) -- never a bare
`sentencepiece.SentencePieceProcessor`, or tokenization is silently wrong.
`KhmerOcrTokenizer` appends `<km>`/`<en>`/`<fr>` language tags and `<eob>`
(end-of-block) past the base vocab; `<PAD>`/`<BOS>`/`<EOS>`/`<UNK>`/`<MASK>`
are the base tokenizer's own reserved ids (looked up dynamically, never
hardcoded). All 4 configured real_data sources are plain, untagged text, so
`encode_plain` (which tags the whole line via a cheap Khmer-vs-ASCII
dominant-language guess) is the only encode path this pipeline exercises;
`encode_tagged` (for data_gen-style inline `<km>...<en>...` spans) is
implemented for forward compatibility but unused here.

```bash
pip install -r recognizer/requirements.txt
python -m recognizer.tokenizer.fetch_tokenizer --out-dir recognizer/tokenizer/assets
```

## Training

```bash
python -m real_data.generate_external_chunks --source all --num-samples 500

# dedup before training -- see real_data/README.md's "Deduplication" section
python -m real_data.deduplicate \
    --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \
                     real_data/samples/chanrith_ocr_image_line \
                     real_data/samples/darayut_scene_text \
                     real_data/samples/sokheng_synthetic_v1 \
    --out-dir real_data/samples/dedup

python -m recognizer.train \
    --dedup-manifest real_data/samples/dedup/manifest.tsv \
    --tokenizer-dir recognizer/tokenizer/assets --run-name v1
```

`--dedup-manifest` (preferred) points at `real_data.deduplicate`'s pooled,
deduplicated output; `--real-data-dirs` (raw per-source manifests, no
dedup pass) is still accepted for quick iteration.

Loss is hybrid CTC (auxiliary, over the full non-windowed encoder output,
`ctc_weight=0.3`) + AR cross-entropy -- the CTC term induces the
left-to-right monotonic image/text correspondence the block-restricted
cross-attention depends on.

### Periodic eval is capped (`max_eval_samples`)

`evaluate_val_cer` decodes val samples **one at a time**, and in
sequential-AR mode emits one token per forward pass. Measured on CUDA:
1072 ms/sample sequential vs 26 ms/sample blockwise. At production scale
(`val_frac=0.02` of 428,911 samples = 8,578 held out) an uncapped
sequential eval is ~153 minutes -- *every* `eval_every=500` steps. That is
not a metric, it is a stall, and it is what an earlier run's "stuck for
hours at step 1000" turned out to be.

The two metrics share one `model.encode` per sample but are priced very
differently after that: CTC CER is an argmax over that output, while AR CER
is a full greedy decode. So they get **separate caps**:

| Knob | Default | Covers |
|---|---|---|
| `max_eval_samples` (`--max-eval-samples`) | 512 | both metrics' sample pool; `0` = whole val set |
| `max_ar_eval_samples` (`--max-ar-eval-samples`) | 64 | the AR greedy decode only; `0` = CTC CER only |

Measured on CUDA over a 512-sample slice, sequential mode:

```
ar_limit=None (all 512)   380.4s   n_ar=512  ctc_cer=1.0406
ar_limit=64                48.1s   n_ar= 64  ctc_cer=1.0406   <- default
ar_limit=0 (CTC only)       5.8s   n_ar=  0  ctc_cer=1.0406
```

CTC CER is identical in all three -- it always covers the full
`max_eval_samples` pool; only the AR decode is subsampled. Both are fixed
head slices of the already-shuffled val list, so they're unbiased across
sources and constant across the run: the CER curve tracks the model, not
which lines were drawn. The log line reports each metric's own `n` plus the
wall time, so a small-n AR CER is never mistaken for a full-set one:

```
eval step 500 ... val_ar_cer 0.9742 (n=64) val_ctc_cer 0.9574 (n=512 of 8578 held out) [81.2s]
```

### Unlearnable samples are filtered (`filter_unlearnable`)

CTC cannot emit more labels than it has input frames. When a sample's target
is longer, `F.ctc_loss` returns `+inf` -- and `compute_loss` passes
`zero_infinity=True`, which replaces that with `0`. The sample then
contributes **no gradient at all**, silently, for the whole run, while still
costing a forward pass. Nothing in the loss curve reveals it.

Measured on the production dataset (3,000-sample scan):

```
samples where CTC target length exceeds encoder frames: 158/3000 (5.3%)

  deepcopy_khmer_text_recognition     0/903   ( 0.0%)
  darayut_scene_text                  0/737   ( 0.0%)
  chanrith_ocr_image_line             0/687   ( 0.0%)
  sokheng_synthetic_v1              158/673  (23.5%)

worst: 124 target chars against 32 encoder frames from a single 128px chunk
```

`run_training` drops these by default (`--keep-unlearnable` to disable),
after the train/val split so the two stay disjoint, and from **both** so val
CER isn't inflated by lines no model could get right. The scan is
`find_unlearnable`, which reads only image headers -- ~20s over 428k samples.
`encoder_frames_for(width, height, cfg)` derives the frame count from the
header alone; it's verified exact against the real `chunk_line_image`
pipeline on 1,500 real samples.

Note this is the same upstream label noise as the truncation below, hitting
a different mechanism: those transcripts don't match their images, so the AR
path over-generates and the CTC path silently zeroes out.

### AR block truncation is counted, not warned per sample

The uniform block split can hand a block more tokens than
`max_tokens_per_block=8`, and the excess is dropped. On the production
dataset this hits ~5.7% of samples and drops ~12.9% of all AR tokens,
essentially all of it from `sokheng_synthetic_v1` (25.6% of that source),
whose worst cases pair a 1-chunk image with a 50+ token transcript -- i.e.
upstream label noise, not a tuning problem.

`split_into_blocks` warns **once per source** and accumulates the rest into
`dataset.truncation_stats()`, which `train.py` logs alongside each eval.
The message deliberately carries no per-sample numbers: it used to embed
the token count, making every occurrence textually unique, which defeated
`warnings`' own dedup and emitted one warning per affected sample --
enough output to bury the training log and, at Colab's output limits, take
the kernel down with it.

### Logging, checkpoints, resuming

Every `log_every=100` steps, `run_training` reports `step, epoch, loss,
ctc_loss, ce_loss, lr` to **three** places (all under
`checkpoint_root/run_name/`, local disk by default -- see
`recognizer/config.py`'s `DEFAULT_CHECKPOINT_ROOT`):
- **`train.log`** -- timestamped, human-readable, the one meant to be read
  directly (`tail -f checkpoints/<run_name>/train.log` while training runs).
  Also captures auto-batch-size selection, resume events, checkpoint
  saves/pushes, and a training-complete summary -- not just the per-step
  loss line.
- **`train_log.csv`** -- pure numeric, for later plotting/analysis.
- stdout (same lines as `train.log`, via a `logging.StreamHandler`).

`epoch` is estimated as `step / (len(train_samples) // batch_size)` and
logged alongside `step` specifically so you can tell whether you're on
your first pass over the data or the 50th (small local pulls repeat many
times over a long `max_steps` run -- see the epoch-size discussion this
project's chat history covers, if you have it).

A full checkpoint (model + optimizer + scheduler state + `model_cfg`) is
saved locally every `ckpt_every=10_000` steps to
`checkpoint_root/run_name/step_<N>.pt`, and `last.pt` is overwritten every
time too. **To continue training from a local checkpoint**, pass
`--resume checkpoints/<run_name>/last.pt` (or a specific `step_<N>.pt`) --
this restores `step`/model/optimizer/scheduler state exactly, so training
picks up where it left off rather than restarting the LR schedule or
losing optimizer momentum:

```bash
python -m recognizer.train --dedup-manifest real_data/samples/dedup/manifest.tsv \
    --tokenizer-dir recognizer/tokenizer/assets --run-name v1 \
    --resume recognizer/checkpoints/v1/last.pt
```

For Colab/Kaggle/local with adaptive batch sizing and automatic checkpoint
push to the `Panhapich/tuna-ocr` HF repo, use
`notebooks/train_recognizer.ipynb` instead of the bare CLI -- see its first
cell for platform setup (Drive mount, HF secret name: `HF_TOKEN`).

### CPU / GPU / TPU

The training device is auto-detected (`env_utils.get_torch_device()`) --
CLI and notebook both pick it up with no flag needed. Call
`env_utils.describe_accelerator()` (the notebook's section 1 does) to print
the resolved device before committing to a long run.

`detect_accelerator()` checks CUDA first -- a cheap, unambiguous query, and
no Colab/Kaggle runtime offers both -- then probes for a TPU in increasing
order of cost: `PJRT_DEVICE=TPU` (compared by *value*: it is also set to
`CUDA`/`CPU`), then the `TPU_*`/`COLAB_TPU_ADDR`/`XRT_TPU_CONFIG` env vars,
then the `/dev/accel*` device nodes, then `torch_xla.runtime.device_type()`
if torch_xla is installed. Both TPU generations are covered on purpose:
checking only the XRT-era vars (`COLAB_TPU_ADDR`/`XRT_TPU_CONFIG`) misses
every current PJRT runtime, which then silently trains on CPU.

- **GPU**: batch size is auto-probed to fit available VRAM
  (`--auto-batch-size` / the notebook's `auto_batch_size=True`) by trying a
  candidate size and halving on CUDA OOM.
- **TPU**: select the TPU runtime/accelerator on Colab or Kaggle *before*
  starting (both ship `torch_xla` preinstalled there -- no extra install
  needed). Training uses `xm.optimizer_step`/`xm.save` instead of the plain
  `optimizer.step`/`torch.save` calls. The VRAM auto-probe is CUDA-only
  (XLA compiles lazily and doesn't surface a catchable Python OOM); on
  TPU/CPU the configured `batch_size` is used as-is, and asking for
  auto-sizing anyway just logs that and moves on. **Caveats**:
  PyTorch/XLA's CTC op support has historically been inconsistent across
  versions -- if `compute_loss`'s `F.ctc_loss` call errors or is
  unexpectedly slow on your TPU runtime, check for a CPU fallback before
  assuming it's a bug in this repo. And because batches are variable-shaped
  by design (width bucketing), XLA compiles one graph per distinct shape:
  expect a few hundred recompiles before the cache covers the common shapes.

`chunks_per_line` is deliberately left on the host by `move_batch` rather
than shipped to the accelerator with the rest of the batch: it only drives
Python control flow (`.tolist()` in `ConformerEncoder.forward`, `.max()` in
the decode paths), and reading it back from the device would force a
host sync every step -- which on XLA serializes the whole asynchronous
execution pipeline, and on CUDA is a needless stall.

## Evaluation / inference

```bash
python -m recognizer.evaluate --checkpoint recognizer/checkpoints/v1/last.pt \
    --real-data-dirs real_data/samples/darayut_scene_text --tokenizer-dir recognizer/tokenizer/assets
python -m recognizer.infer --checkpoint recognizer/checkpoints/v1/last.pt \
    --image path/to/line.jpg --tokenizer-dir recognizer/tokenizer/assets
```

Greedy decoding only; metric is Character Error Rate (WER would partly
measure Khmer word-segmentation-heuristic agreement, not recognition
accuracy, since Khmer word boundaries are themselves ambiguous).

## Known gaps / next steps

- Relative-position MHSA (full Conformer-paper style) -- this pass uses
  absolute sinusoidal positional encoding, a deliberate v1 simplification.
- Beam search decoding -- greedy only for now.
- The uniform-interval `<eob>` block-boundary split (no real per-token
  alignment exists in any source) could be refined via CTC-based
  re-alignment instead.
- `decode_greedy`/`decode_block` recompute the whole decoder prefix on
  every block step (no KV-cache) -- fine for short OCR lines, but an
  optimization opportunity for longer sequences.
- No distributed training, tuned mixed precision, or hyperparameter search
  -- out of scope for this pass. A bare `--amp` flag may be added cheaply
  later if needed.
- TPU support runs on a single core only -- a multi-core TPU (e.g. Colab's
  v2-8/v3-8) is not parallelized across cores (`xmp.spawn` / multi-core
  `MpDeviceLoader`), so most of an 8-core TPU's throughput goes unused.
  Follow-up work, not required for this to run correctly on TPU today.
