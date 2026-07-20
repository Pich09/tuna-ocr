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
python -m recognizer.train \
    --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \
                     real_data/samples/chanrith_ocr_image_line \
                     real_data/samples/darayut_scene_text \
                     real_data/samples/sokheng_synthetic_v1 \
    --tokenizer-dir recognizer/tokenizer/assets --run-name v1
```

Loss is hybrid CTC (auxiliary, over the full non-windowed encoder output,
`ctc_weight=0.3`) + AR cross-entropy -- the CTC term induces the
left-to-right monotonic image/text correspondence the block-restricted
cross-attention depends on. Logs `step, loss, ctc_loss, ce_loss, lr` every
`log_every=100` steps (stdout + `checkpoints/run_name/train_log.csv`); saves
a full checkpoint every `ckpt_every=10_000` steps.

For Colab/Kaggle/local with adaptive batch sizing and automatic checkpoint
push to the `Panhapich/tuna-ocr` HF repo, use
`notebooks/train_recognizer.ipynb` instead of the bare CLI -- see its first
cell for platform setup (Drive mount, HF secret name: `HF_TOKEN`).

### CPU / GPU / TPU

The training device is auto-detected (`env_utils.get_torch_device()`) --
CLI and notebook both pick it up with no flag needed:
- **GPU**: batch size is auto-probed to fit available VRAM
  (`--auto-batch-size` / the notebook's `auto_batch_size=True`) by trying a
  candidate size and halving on CUDA OOM.
- **TPU**: select the TPU runtime/accelerator on Colab or Kaggle *before*
  starting (both ship `torch_xla` preinstalled there -- no extra install
  needed). Training uses `xm.optimizer_step`/`xm.save` instead of the plain
  `optimizer.step`/`torch.save` calls. The VRAM auto-probe is skipped on TPU
  (XLA doesn't surface a catchable Python OOM the same way); the configured
  `batch_size` is used as-is. **Caveat**: PyTorch/XLA's CTC op support has
  historically been inconsistent across versions -- if `compute_loss`'s
  `F.ctc_loss` call errors or is unexpectedly slow on your TPU runtime,
  check for a CPU fallback before assuming it's a bug in this repo.

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
