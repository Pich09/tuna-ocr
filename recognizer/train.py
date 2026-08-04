"""Training loop for the Recognizer (Conformer encoder + blockwise-AR
decoder), over real_data-only sources (see recognizer/config.py /
recognizer/data/manifest.py -- data_gen synthetic output is out of scope).

Run `real_data.deduplicate` first and pass its output via --dedup-manifest
(preferred) so training never sees exact/near-duplicate images across
sources; --real-data-dirs (raw, undeduplicated per-source manifests) is
still supported for quick iteration but skips that dedup pass.

CLI:
    python -m real_data.deduplicate --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \\
        real_data/samples/chanrith_ocr_image_line real_data/samples/darayut_scene_text \\
        real_data/samples/sokheng_synthetic_v1 --out-dir real_data/samples/dedup

    python -m recognizer.train --dedup-manifest real_data/samples/dedup/manifest.tsv \\
        --tokenizer-dir recognizer/tokenizer/assets --run-name v1

Runs on CPU, CUDA, or TPU -- the device is auto-detected (see env_utils.py)
unless explicitly passed. TPU note: PyTorch/XLA's CTC op support has
historically been spotty across versions; if `compute_loss`'s CTCLoss call
errors or silently falls back to a slow CPU path on your TPU runtime,
that's a torch_xla limitation, not a bug in this file -- report the exact
error rather than assuming it's this code.

Also exposes `run_training(...)` as a plain callable so the Colab/Kaggle/
local notebook (notebooks/train_recognizer.ipynb) can drive training
directly instead of shelling out to this CLI.
"""
import argparse
import csv
import logging
import math
import os
import random
import time
from pathlib import Path

# Must run before `import torch` below (and before anything else in this process
# ever imports torch) -- the CUDA allocator reads its config once, at first
# CUDA-context init, and setting this any later has no effect. setdefault, not
# assignment, so a caller who already set it (e.g. a notebook's very first cell,
# the only place this can reliably run before the notebook's OWN earlier `import
# torch` cell) isn't silently overridden. Targets allocator fragmentation from
# many steps of variable-shaped batches (width-bucketed training -> a different
# ar_logits/enc_out size nearly every step): observed in practice as a CUDA OOM
# ~47k steps into an otherwise-healthy run, with "reserved but unallocated"
# memory in the traceback -- PyTorch's own suggested fix for exactly that
# signature. Both env var names are set since different torch versions read
# different ones (newer: PYTORCH_ALLOC_CONF; CUDA-specific alias some versions
# still expect: PYTORCH_CUDA_ALLOC_CONF) -- harmless if a given build ignores one.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import env_utils
from .config import ModelConfig, TrainConfig, DEFAULT_CHECKPOINT_ROOT, TOKENIZER_ASSETS_DIR
from .data.char_vocab import CharVocab, build_char_vocab
from .data.dataset import (BucketBatchSampler, OCRLineDataset, compute_widths, make_collate_fn,
                           find_unlearnable, move_batch, truncation_stats)
from .data.manifest import build_combined_index, load_dedup_manifest
from .data.transforms import chunk_line_image
from .hf_push import push_checkpoint
from .modules.decoder import truncate_at_eos, truncate_blocks
from .modules.model import Recognizer
from .tokenizer.khmer_ocr_tokenizer import KhmerOcrTokenizer


def build_logger(ckpt_dir: Path, run_name: str) -> logging.Logger:
    """Every training-status line (auto batch size, resume, per-step
    metrics, checkpoint saves/pushes) goes through this logger instead of
    bare print(), so it's captured in `ckpt_dir/train.log` (timestamped,
    human-readable) as well as stdout -- print() alone is lost the moment
    the terminal/notebook session closes. `train_log.csv` (written
    separately, see run_training) stays purely numeric for later plotting;
    this .log file is the one meant to be read directly."""
    logger = logging.getLogger(f"recognizer.train.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # re-entrant safe: run_training may be called more than once per process (e.g. notebook)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(ckpt_dir / "train.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def find_max_batch_size(model, dataset, collate_fn, device, widths, start: int = 64, min_batch: int = 1) -> int:
    """OOM-probing auto-tune: try `start`, halve on CUDA OOM until a batch
    fits (forward + backward). CPU/TPU just use the configured default --
    on TPU, memory errors don't surface as a catchable Python OOM the same
    way (XLA compiles lazily), so this probe isn't meaningful there.

    Probes against the `bs` WIDEST samples in the dataset (by `widths`, one
    entry per dataset index), not arbitrary ones: BucketBatchSampler groups
    similar-width images into the same batch, so the batch of widest images
    has the largest memory footprint of any batch actually seen during
    training. Probing with arbitrary samples can under-estimate that peak
    and pass, only to OOM later mid-training once the width-sorted order
    reaches its widest batch -- losing all progress since the last
    checkpoint (observed in practice: a probe at bs=64 against non-worst-case
    samples passed, then training OOM'd ~5300 steps in on the real widest
    batch).

    Also reserves (and immediately frees) a dummy allocation matching
    AdamW's exp_avg + exp_avg_sq per-parameter state: that state is only
    allocated lazily on the optimizer's first real step(), which happens
    after this probe and outside it, so a plain forward+backward here would
    otherwise under-count real training's steady-state memory by roughly
    2x the model's parameter footprint. A dummy tensor (not a real
    optimizer.step()) verifies the room exists without nudging the model's
    actual initialization with a bogus gradient from this probe's fake
    (sum-of-logits) objective."""
    if device.type != "cuda":
        return start
    widest_idx = sorted(range(len(widths)), key=lambda i: widths[i], reverse=True)
    optim_state_bytes = sum(p.numel() for p in model.parameters()) * 2 * 4  # fp32 exp_avg + exp_avg_sq
    bs = start
    while bs >= min_batch:
        try:
            idxs = widest_idx[:bs]
            batch = collate_fn([dataset[i] for i in idxs])
            batch = move_batch(batch, device)
            ctc_logits, ar_logits, _ = model(batch["chunks"], batch["chunks_per_line"], batch["ar_targets"])
            (ctc_logits.float().sum() + ar_logits.float().sum()).backward()
            optim_reserve = torch.empty(optim_state_bytes, dtype=torch.uint8, device=device)
            del optim_reserve
            model.zero_grad(set_to_none=True)
            del ctc_logits, ar_logits
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            bs //= 2
    return max(min_batch, 1)


def lr_lambda(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.0) -> float:
    if step < warmup_steps:
        linear = step / max(1, warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * linear
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def compute_loss(ctc_logits, ar_logits, batch, ctc_weight: float, pad_id: int, ctc_blank_id: int,
                 enc_lengths=None, ar_targets=None):
    b, t_max, _ = ctc_logits.shape
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T,N,C)
    # Real per-sample encoder lengths, not t_max: enc_out is zero-padded up to
    # the batch's longest line, and claiming those padding frames are real
    # input makes CTC align targets against blank padding.
    if enc_lengths is not None:
        ctc_input_lengths = enc_lengths.to(dtype=torch.long, device=ctc_logits.device)
    else:
        ctc_input_lengths = torch.full((b,), t_max, dtype=torch.long, device=ctc_logits.device)
    # CTCLoss expects the blank id to match the logits' extra class; ctc_head
    # appends it at index vocab_size, so remap that here explicitly.
    ctc_loss = F.ctc_loss(
        log_probs, batch["ctc_targets"], ctc_input_lengths, batch["ctc_lengths"],
        blank=ctc_blank_id, zero_infinity=True,
    )
    # ar_targets: whichever target tensor matches ar_logits' shape for the mode this
    # step ran in -- (B, num_blocks, K) for blockwise, (B, L) flat for sequential (see
    # run_training's ar_mode/ar_targets_for_mode). Defaults to batch["ar_targets"] for
    # callers that don't pass it explicitly (e.g. find_max_batch_size's OOM probe,
    # which always probes blockwise mode).
    if ar_targets is None:
        ar_targets = batch["ar_targets"]
    ce_loss = F.cross_entropy(
        ar_logits.reshape(-1, ar_logits.shape[-1]), ar_targets.reshape(-1),
        ignore_index=pad_id,
    )
    loss = ctc_weight * ctc_loss + (1 - ctc_weight) * ce_loss
    return loss, ctc_loss, ce_loss


@torch.no_grad()
def log_inference_samples(model, tokenizer, model_cfg, samples, device, logger, step, char_vocab=None,
                          mode: str = "blockwise"):
    """Greedy-decodes a handful of fixed held-out samples and logs
    prediction vs ground truth -- a cheap, human-readable sanity check that
    the model is actually learning to read (not just that the loss number
    is dropping), run periodically alongside the numeric metrics."""
    was_training = model.training
    model.eval()
    for i, sample in enumerate(samples):
        chunk_tensors, _ = chunk_line_image(
            sample.image_source, model_cfg.chunk_width, model_cfg.chunk_overlap, model_cfg.img_height,
        )
        chunk_batch = torch.stack(chunk_tensors).to(device)
        chunks_per_line = torch.tensor([len(chunk_tensors)], dtype=torch.long)
        enc_out, enc_lengths, enc_block_ids = model.encode(chunk_batch, chunks_per_line)
        if mode == "sequential":
            tokens = model.decoder.decode_greedy_sequential(enc_out, enc_lengths)[0].tolist()
            tokens = truncate_at_eos(tokens, tokenizer.eos_id, tokenizer.pad_id)
        else:
            max_blocks = min(len(chunk_tensors), 256 // max(1, model_cfg.max_tokens_per_block) + 1)
            tokens = model.decoder.decode_greedy(enc_out, enc_lengths, enc_block_ids, max_blocks)[0].tolist()
            tokens = truncate_blocks(tokens, model_cfg.max_tokens_per_block,
                                     tokenizer.eob_id, tokenizer.pad_id, num_blocks=len(chunk_tensors))
        pred = tokenizer.decode(tokens, strip_control=True)
        # CTC greedy is the encoder's own read-out: it is alignment-free and
        # not exposed to the decoder's exposure bias, so it is the earliest
        # honest signal that the encoder is learning to SEE text at all.
        ctc_txt = ""
        if char_vocab is not None:
            # One device->host transfer for the whole row -- indexing the GPU
            # tensor per frame costs a sync per frame.
            n = int(enc_lengths[0])
            row = model.ctc_head(enc_out).argmax(-1)[0][:n].cpu().tolist()
            ctc_txt = char_vocab.decode(_collapse_ctc(row, char_vocab.blank_id))
        logger.info(f"  [sample {i}] step {step} gt={sample.text!r} ctc={ctc_txt!r} ar={pred!r}")
    if was_training:
        model.train()


def _collapse_ctc(row: list, blank_id: int) -> list:
    """Standard CTC greedy post-processing: collapse repeats, drop blanks."""
    out, prev = [], None
    for c in row:
        if c != prev and c != blank_id:
            out.append(c)
        prev = c
    return out


def _scaled_width(sample, img_height: int) -> int:
    """Width the line will have after the fixed-height resize -- what actually
    determines its chunk count (see data/dataset.py's compute_widths)."""
    from .data.transforms import open_image  # noqa: PLC0415
    with open_image(sample.image_source) as img:
        w, h = img.size
    return max(1, round(w * img_height / max(1, h)))


def evaluate_val_cer(model, tokenizer, model_cfg, val_samples, device, char_vocab, mode: str = "blockwise",
                     ar_limit: int = None, batch_size: int = 16):
    """Greedy-decodes held-out samples and returns
    (ar_cer, ctc_cer, n_ar, n_ctc) -- real quantitative numbers to track
    alongside the loss (loss can keep falling on train while val CER climbs;
    that gap is the overfitting signal). AR CER is the decoder's read-out,
    CTC CER the encoder's alignment-free read-out. Runs in eval mode,
    restores train mode on exit; no grad.

    The two metrics are priced very differently and so are sampled
    differently: the CTC read-out is an argmax over an encoder pass, while
    the AR read-out is a full greedy decode -- in "sequential" mode one
    token per forward pass. `ar_limit` caps the AR decode at the FIRST N
    samples (a fixed subset, so the CER curve tracks the model, not the
    draw) while CTC CER covers all of `val_samples`; the returned counts say
    what each number was measured over.

    Decoding is batched (`batch_size` lines per encoder pass, sorted by
    scaled width within each metric's pool to limit padding -- CER over a
    set is order-independent, so sorting is free) and every hypothesis is
    truncated at its real end: sequential rows at their first <eos>/<pad>,
    blockwise rows per block at <eob>/<pad> and at the line's own block
    count. batch_size=16, not larger: the sequential decode's cross-attention
    grows with batch x prefix x encoder frames, and at 32 the widest-line
    batches thrashed a 4GB GPU (188s vs 37s at 16 for the same 64 samples). Without the truncation, hypotheses carried up to max_len=256
    tokens of post-<eos> garbage -- the source of the impossible val_ar_cer
    values above 1.0 in earlier runs."""
    import editdistance  # noqa: PLC0415 -- kept local so training without eval has no hard dep

    def cer(refs, hyps):
        if not refs:
            return float("nan")
        edits = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
        return edits / (sum(len(r) for r in refs) or 1)

    def encode_batch(batch_samples):
        all_chunks, cpl = [], []
        for s in batch_samples:
            ct, _ = chunk_line_image(
                s.image_source, model_cfg.chunk_width, model_cfg.chunk_overlap, model_cfg.img_height)
            all_chunks.extend(ct)
            cpl.append(len(ct))
        chunk_batch = torch.stack(all_chunks).to(device)
        # host tensor on purpose -- the encoder only reads this via .tolist();
        # see data/dataset.py's HOST_ONLY_BATCH_KEYS.
        enc = model.encode(chunk_batch, torch.tensor(cpl, dtype=torch.long))
        return enc, cpl

    def batches(sample_list):
        ordered = sorted(sample_list, key=lambda s: _scaled_width(s, model_cfg.img_height))
        for i in range(0, len(ordered), batch_size):
            yield ordered[i:i + batch_size]

    was_training = model.training
    model.eval()

    ctc_refs, ctc_hyps = [], []
    ar_refs, ar_hyps = [], []
    ar_samples = list(val_samples) if ar_limit is None else list(val_samples[:ar_limit])

    with torch.no_grad():
        # CTC CER over the full pool: argmax + collapse, one transfer per batch.
        for group in batches(val_samples):
            (enc_out, enc_lengths, _), _ = encode_batch(group)
            ids = model.ctc_head(enc_out).argmax(-1).cpu()
            lengths = enc_lengths.cpu().tolist()
            for sample, row, n in zip(group, ids.tolist(), lengths):
                ctc_refs.append(sample.text)
                ctc_hyps.append(char_vocab.decode(_collapse_ctc(row[:n], char_vocab.blank_id)))

        # AR CER over the capped pool (re-encodes those lines; the encoder
        # pass is ~26ms/sample, noise next to the decode it feeds).
        for group in batches(ar_samples):
            (enc_out, enc_lengths, enc_block_ids), cpl = encode_batch(group)
            if mode == "sequential":
                rows = model.decoder.decode_greedy_sequential(enc_out, enc_lengths).tolist()
                trimmed = [truncate_at_eos(r, tokenizer.eos_id, tokenizer.pad_id) for r in rows]
            else:
                max_blocks = min(max(cpl), 256 // max(1, model_cfg.max_tokens_per_block) + 1)
                rows = model.decoder.decode_greedy(enc_out, enc_lengths, enc_block_ids, max_blocks).tolist()
                trimmed = [
                    truncate_blocks(r, model_cfg.max_tokens_per_block, tokenizer.eob_id,
                                    tokenizer.pad_id, num_blocks=n)
                    for r, n in zip(rows, cpl)
                ]
            for sample, tokens in zip(group, trimmed):
                ar_refs.append(sample.text)
                ar_hyps.append(tokenizer.decode(tokens, strip_control=True))

    if was_training:
        model.train()
    return cer(ar_refs, ar_hyps), cer(ctc_refs, ctc_hyps), len(ar_refs), len(ctc_refs)


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig, real_data_roots=None, dedup_manifest_path=None,
                  tokenizer_dir=TOKENIZER_ASSETS_DIR, checkpoint_root=DEFAULT_CHECKPOINT_ROOT, run_name: str = "v1",
                  push_to_hub: bool = False, repo_id=None, hf_token=None, hub_private: bool = True,
                  auto_batch_size: bool = False, resume_path=None, device=None):
    """Exactly one of `dedup_manifest_path` (preferred -- output of
    `real_data.deduplicate`) or `real_data_roots` (raw, undeduplicated
    per-source manifests) must be given."""
    device = device or env_utils.get_torch_device()
    is_xla = device.type == "xla"
    xm = None
    if is_xla:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
    torch.manual_seed(train_cfg.seed)

    # print()s (not logger -- the logger isn't built until further down, once
    # ckpt_dir exists) with flush=True at each stage up through the first real
    # logger.info() call: this whole preamble ran in a handful of seconds
    # against the real ~430k-row production dataset when benchmarked directly
    # (load_dedup_manifest ~5-14s, compute_widths ~13s), so if a run is stuck
    # with no output for a long time, these pin down exactly which stage it's
    # actually stuck in, rather than leaving a single opaque multi-step gap.
    print("run_training: loading tokenizer...", flush=True)
    tokenizer = KhmerOcrTokenizer(tokenizer_dir)
    print(f"run_training: loading samples from {dedup_manifest_path or real_data_roots}...", flush=True)
    if dedup_manifest_path:
        samples = load_dedup_manifest(dedup_manifest_path)
    elif real_data_roots:
        samples = build_combined_index(real_data_roots)
    else:
        raise ValueError("run_training needs either dedup_manifest_path or real_data_roots")
    if not samples:
        raise RuntimeError(f"No samples found (dedup_manifest_path={dedup_manifest_path}, "
                            f"real_data_roots={real_data_roots}) -- generate/deduplicate data first with "
                            f"`python -m real_data.generate_external_chunks` and `python -m real_data.deduplicate`.")
    print(f"run_training: loaded {len(samples)} samples, shuffling/splitting...", flush=True)
    # Shuffle before splitting: the pooled manifest is written source-by-
    # source, so a head slice would make the whole val set (and every sampled
    # prediction) come from one source only.
    samples = list(samples)
    random.Random(train_cfg.seed).shuffle(samples)
    n_val = max(1, int(len(samples) * train_cfg.val_frac))
    train_samples, val_samples = samples[n_val:], samples[:n_val]

    ckpt_dir_early = Path(checkpoint_root) / run_name
    ckpt_dir_early.mkdir(parents=True, exist_ok=True)
    char_vocab_path = ckpt_dir_early / "char_vocab.json"
    print(f"run_training: {'loading' if char_vocab_path.exists() else 'building'} char vocab...", flush=True)
    if char_vocab_path.exists():
        char_vocab = CharVocab.load(char_vocab_path)
    else:
        char_vocab = build_char_vocab((s.text for s in train_samples), min_count=2)
        char_vocab.save(char_vocab_path)

    # Drop samples CTC provably cannot fit (target longer than the encoder
    # frames the image yields). Their loss is +inf, which zero_infinity=True
    # turns into 0, so they contribute nothing but still cost a forward pass --
    # and in sequential-AR mode the decoder is meanwhile taught to emit their
    # full impossible transcript, which is worse than nothing. Runs after the
    # split so train/val stay disjoint, and on both so val CER isn't inflated
    # by lines no model could get right.
    if train_cfg.filter_unlearnable:
        print("run_training: scanning for unlearnable samples (CTC target > encoder frames)...", flush=True)
        t_f = time.time()
        n_before = len(train_samples), len(val_samples)
        dropped_by_source = {}
        for name, lst in (("train", train_samples), ("val", val_samples)):
            bad = set(find_unlearnable(lst, model_cfg, char_vocab))
            for i in bad:
                src = lst[i].source
                dropped_by_source[src] = dropped_by_source.get(src, 0) + 1
            lst[:] = [s for i, s in enumerate(lst) if i not in bad]
        print(f"run_training: dropped {n_before[0] - len(train_samples)} train / "
              f"{n_before[1] - len(val_samples)} val unlearnable samples in {time.time() - t_f:.1f}s "
              f"({dropped_by_source or 'none'})", flush=True)
        if not train_samples:
            raise RuntimeError("Every training sample was filtered as unlearnable -- check "
                               "model_cfg.chunk_width/img_height against your data before rerunning.")

    # fixed subset (not re-sampled) so the same lines are tracked across the
    # whole run -- otherwise a changing sample set would confound "is this
    # getting more readable" with "is this a different, easier/harder line".
    inference_samples = val_samples[:train_cfg.num_samples]
    # Head slice, not a re-sample: val_samples is already shuffled, so the head
    # is an unbiased draw across sources, and holding it FIXED means the CER
    # curve tracks the model rather than which lines happened to be drawn.
    eval_samples = (val_samples[:train_cfg.max_eval_samples]
                    if train_cfg.max_eval_samples else val_samples)

    train_ds = OCRLineDataset(train_samples, tokenizer, model_cfg, char_vocab=char_vocab)
    collate_fn = make_collate_fn(tokenizer, model_cfg.chunk_width)

    print(f"run_training: building model + moving to {device}...", flush=True)
    model = Recognizer(model_cfg, tokenizer.vocab_size, bos_id=tokenizer.bos_id,
                       pad_id=tokenizer.pad_id, ctc_vocab_size=char_vocab.size,
                       eos_id=tokenizer.eos_id, eob_id=tokenizer.eob_id).to(device)

    ckpt_dir = ckpt_dir_early
    logger = build_logger(ckpt_dir, run_name)
    logger.info(f"run '{run_name}': device={device}, {len(train_samples)} train / {len(val_samples)} val samples, "
                f"checkpoints -> {ckpt_dir}, log file -> {ckpt_dir / 'train.log'}")

    # Shared by the OOM probe (worst-case batch) and the bucket sampler.
    # Cached next to the data it describes, so the multi-minute width scan is
    # paid once per dataset rather than on every restart.
    widths_cache = (Path(dedup_manifest_path).parent / "widths_cache.json"
                    if dedup_manifest_path else ckpt_dir / "widths_cache.json")
    t_w = time.time()
    widths = compute_widths(train_ds, cache_path=widths_cache)
    logger.info(f"image widths ready in {time.time() - t_w:.1f}s (cache: {widths_cache})")

    batch_size = train_cfg.batch_size
    if auto_batch_size and device.type == "cuda":
        batch_size = find_max_batch_size(model, train_ds, collate_fn, device, widths, start=max(64, train_cfg.batch_size))
        logger.info(f"auto batch size: {batch_size}")
    elif auto_batch_size:
        # Gated on CUDA at the call site, not just inside the probe: the probe's
        # non-CUDA early-return hands back its `start` argument, which is
        # max(64, batch_size) -- i.e. asking for auto-sizing on TPU/CPU used to
        # silently *raise* the batch size to 64 rather than leave the config
        # alone, which is the opposite of the documented "just use the
        # configured default" behaviour.
        logger.info(f"auto batch size requested but device is {device.type} (OOM probing is "
                    f"CUDA-only -- XLA compiles lazily and doesn't raise a catchable Python "
                    f"OOM); using the configured batch_size={batch_size}")

    steps_per_epoch = max(1, len(train_ds) // batch_size)
    logger.info(f"batch_size={batch_size}, ~{steps_per_epoch} steps/epoch, max_steps={train_cfg.max_steps} "
                f"(~{train_cfg.max_steps / steps_per_epoch:.1f} epochs)")

    # Guard against warmup >= max_steps: the LR schedule is linear-warmup then
    # cosine-decay, so if warmup_steps >= max_steps the run *never leaves
    # warmup* -- LR ramps but never reaches peak and never decays, so learning
    # crawls the whole run (this is exactly what stalled CTC in an earlier run:
    # warmup_steps=4000 default paired with an overridden max_steps=3000 left LR
    # at ~8% of peak through the phase where CTC should learn to spike, so it
    # sat at all-blank). Clamp to 10% of max_steps and log loudly rather than
    # silently crippling the run.
    if train_cfg.warmup_steps >= train_cfg.max_steps:
        clamped = max(1, int(0.1 * train_cfg.max_steps))
        logger.info(f"WARNING: warmup_steps={train_cfg.warmup_steps} >= max_steps={train_cfg.max_steps} "
                    f"-> LR would never reach peak; clamping warmup_steps to {clamped} (10% of max_steps).")
        train_cfg.warmup_steps = clamped

    sampler = BucketBatchSampler(train_ds, batch_size=batch_size, shuffle=True, widths=widths)
    # DataLoader workers must use the "fork" start method: the dataset holds a
    # KhmerOcrTokenizer whose `_base` is a *dynamically imported*
    # khmer_segmentation.KhmerTokenizer (loaded from tokenizer assets, not an
    # importable module), which is unpicklable. Python 3.14 defaults to the
    # "forkserver"/"spawn" start method on Linux, which pickles the dataset to
    # hand it to each worker -> PicklingError before step 1. "fork" instead
    # inherits parent memory, so nothing is pickled. Workers only do CPU-bound
    # image decode (never touch CUDA), so forking after CUDA init in the parent
    # is safe here.
    if train_cfg.num_workers > 0:
        import multiprocessing as mp  # noqa: PLC0415
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    # num_workers>0 so image loading/decoding for the *next* batch overlaps
    # with the GPU forward/backward of the *current* one -- without this the
    # DataLoader runs synchronously on the main process and the GPU sits
    # idle between steps waiting on CPU-bound JPEG decode, which is exactly
    # the "GPU not fully utilized" symptom this was tuned to fix.
    loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_fn,
        num_workers=train_cfg.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=train_cfg.num_workers > 0,
        prefetch_factor=4 if train_cfg.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    min_lr_ratio = train_cfg.min_lr / train_cfg.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, train_cfg.warmup_steps, train_cfg.max_steps, min_lr_ratio),
    )

    log_path = ckpt_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,epoch,loss,ctc_loss,ce_loss,lr\n")

    eval_log_path = ckpt_dir / "eval_log.csv"
    if not eval_log_path.exists():
        eval_log_path.write_text("step,epoch,val_ar_cer,val_ctc_cer\n")

    step = 0
    if resume_path:
        # weights_only=False: this checkpoint is self-produced (stores our own
        # ModelConfig dataclass alongside the state dicts), not an untrusted
        # third-party file -- PyTorch 2.6+'s weights_only=True default would
        # otherwise reject the ModelConfig global. map_location="cpu": XLA
        # devices aren't reliably understood by map_location -- load_state_dict
        # onto the already-.to(device)'d model handles the actual placement.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        step = state["step"]
        logger.info(f"resumed from {resume_path} at step {step} "
                    f"({train_cfg.max_steps - step} steps remaining to max_steps={train_cfg.max_steps})")

    ctc_blank_id = char_vocab.blank_id
    if train_cfg.sequential_ar_steps > 0:
        logger.info(f"AR decoder training curriculum: sequential mode for steps 0-"
                    f"{train_cfg.sequential_ar_steps} (plain teacher forcing, unrestricted "
                    f"cross-attention -- see modules/decoder.py), then blockwise for the rest.")
    ar_mode_logged_switch = train_cfg.sequential_ar_steps <= 0  # already "switched" if disabled from the start
    oom_skips = 0  # see the except block below -- a real run hit a CUDA OOM ~47k steps
                   # in with no protection past the one-time startup batch-size probe,
                   # killing the entire multi-hour run over a single bad batch.
    model.train()
    t0 = time.time()
    while step < train_cfg.max_steps:
        for batch in loader:
            if step >= train_cfg.max_steps:
                break
            ar_mode = "sequential" if step < train_cfg.sequential_ar_steps else "blockwise"
            if ar_mode == "blockwise" and not ar_mode_logged_switch:
                logger.info(f"step {step}: switching AR decoder from sequential to blockwise mode")
                ar_mode_logged_switch = True
            try:
                batch = move_batch(batch, device)
                ar_targets_for_mode = batch["ar_flat_targets"] if ar_mode == "sequential" else batch["ar_targets"]
                ctc_logits, ar_logits, enc_lengths = model(
                    batch["chunks"], batch["chunks_per_line"], ar_targets_for_mode, mode=ar_mode)
                loss, ctc_loss, ce_loss = compute_loss(
                    ctc_logits, ar_logits, batch, train_cfg.ctc_weight, tokenizer.pad_id, ctc_blank_id,
                    enc_lengths=enc_lengths, ar_targets=ar_targets_for_mode,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if is_xla:
                    xm.optimizer_step(optimizer)  # reduces gradients across cores + steps + marks the XLA graph
                else:
                    optimizer.step()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # torch.cuda.OutOfMemoryError is the clean case, but NOT the only shape
                # a memory failure takes here: with PYTORCH_ALLOC_CONF=expandable_segments
                # (set near this module's imports, specifically to prevent OOMs) an
                # over-budget CUDA allocation was observed to instead raise a plain
                # RuntimeError("CUDA driver error: device not ready") on at least one
                # driver/GPU combination -- confirmed empty_cache() and further CUDA work
                # both still succeed afterward in the same process, so this is exactly as
                # recoverable as the clean case, just wrapped differently. A bare
                # RuntimeError can mean many things that AREN'T memory-related (a real
                # bug, a shape mismatch), so only the OOM-shaped ones are swallowed here;
                # anything else re-raises immediately rather than silently skipping steps
                # over a genuine bug.
                msg = str(exc).lower()
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                    s in msg for s in ("out of memory", "device not ready", "cuda driver error"))
                if not is_oom:
                    raise
                # Observed cause in practice: CUDA allocator fragmentation building up
                # over many steps of variable-shaped batches (width-bucketed training --
                # a different ar_logits/enc_out size nearly every step), not a genuine
                # peak-memory miss on this specific batch -- so skipping the one unlucky
                # batch and continuing is the right response, not shrinking the batch
                # size forever the way the startup probe does.
                oom_skips += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                logger.warning(
                    f"step {step}: CUDA OOM on a batch of {batch['chunks_per_line'].shape[0]} lines / "
                    f"{batch['chunks'].shape[0]} chunks (mode={ar_mode}, oom_skips so far: {oom_skips}) "
                    f"-- skipping this batch and continuing. {exc}"
                )
                continue
            scheduler.step()
            step += 1

            if step % train_cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                epoch = step / steps_per_epoch
                oom_note = f" oom_skips {oom_skips}" if oom_skips else ""
                logger.info(f"step {step} epoch {epoch:.2f} loss {loss.item():.4f} ctc {ctc_loss.item():.4f} "
                            f"ce {ce_loss.item():.4f} lr {lr:.2e} ({elapsed:.1f}s){oom_note}")
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow([step, f"{epoch:.4f}", loss.item(), ctc_loss.item(), ce_loss.item(), lr])

            if train_cfg.sample_every > 0 and step % train_cfg.sample_every == 0 and inference_samples:
                log_inference_samples(model, tokenizer, model_cfg, inference_samples, device, logger, step,
                                      char_vocab=char_vocab, mode=ar_mode)

            if train_cfg.eval_every > 0 and step % train_cfg.eval_every == 0 and eval_samples:
                t_eval = time.time()
                ar_cer, ctc_cer, n_ar, n_ctc = evaluate_val_cer(
                    model, tokenizer, model_cfg, eval_samples, device, char_vocab, mode=ar_mode,
                    ar_limit=train_cfg.max_ar_eval_samples)
                epoch = step / steps_per_epoch
                logger.info(f"eval step {step} epoch {epoch:.2f} val_ar_cer {ar_cer:.4f} (n={n_ar}) "
                            f"val_ctc_cer {ctc_cer:.4f} (n={n_ctc} of {len(val_samples)} held out) "
                            f"[{time.time() - t_eval:.1f}s]")
                trunc = truncation_stats()
                if trunc:
                    logger.info("AR block truncation so far: " + ", ".join(
                        f"{src}: {n} blocks / {dropped} tokens dropped" for src, (n, dropped) in trunc.items()))
                with eval_log_path.open("a", newline="") as f:
                    csv.writer(f).writerow([step, f"{epoch:.4f}", f"{ar_cer:.4f}", f"{ctc_cer:.4f}"])

            if step % train_cfg.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                state = {
                    "step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "model_cfg": model_cfg,
                    "char_vocab": char_vocab.to_json(),
                }
                save_fn = xm.save if is_xla else torch.save  # xm.save moves XLA tensors to CPU before writing
                save_fn(state, ckpt_path)
                save_fn(state, ckpt_dir / "last.pt")
                logger.info(f"saved checkpoint {ckpt_path} (and {ckpt_dir / 'last.pt'} -- "
                            f"resume from either with --resume)")
                if push_to_hub:
                    kwargs = {"token": hf_token, "private": hub_private}
                    if repo_id:
                        kwargs["repo_id"] = repo_id
                    url = push_checkpoint(ckpt_path, **kwargs)
                    logger.info(f"pushed checkpoint to {url}")

    oom_note = f", {oom_skips} batches skipped due to CUDA OOM" if oom_skips else ""
    logger.info(f"training complete: {step} steps, checkpoints + log in {ckpt_dir}{oom_note}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dedup-manifest", type=Path,
                             help="Output of `real_data.deduplicate` -- preferred over --real-data-dirs.")
    data_group.add_argument("--real-data-dirs", nargs="+",
                             help="Raw, undeduplicated per-source manifests (skips the dedup pass).")
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_ASSETS_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None,
                         help="Linear LR warmup steps before cosine decay. Keep well below max_steps "
                              "(~10 percent) -- if >= max_steps the run never reaches peak LR "
                              "(auto-clamped with a warning). The config default assumes the long max_steps.")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None,
                         help="Greedy-decode a few fixed held-out samples every N steps and log pred vs "
                              "ground truth, as a readability sanity check alongside the loss numbers. "
                              "0 disables.")
    parser.add_argument("--eval-every", type=int, default=None,
                         help="Compute quantitative held-out val CER (AR + CTC) every N steps, logged to "
                              "train.log and eval_log.csv. 0 disables.")
    parser.add_argument("--keep-unlearnable", action="store_true",
                         help="Keep samples whose CTC target is longer than the encoder frames their "
                              "image yields. Those have ctc_loss=+inf, zeroed by zero_infinity, so they "
                              "contribute no gradient; they are dropped by default.")
    parser.add_argument("--max-ar-eval-samples", type=int, default=None,
                         help="Cap on the AR greedy decode inside each periodic eval (default 64). The "
                              "AR decode dominates eval cost (~1072 ms/sample in sequential mode vs ~26 "
                              "blockwise); CTC CER still covers all --max-eval-samples. 0 = CTC only.")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                         help="Cap on val samples decoded per periodic eval (default 512). The full val "
                              "set is thousands of samples decoded one at a time, which at production "
                              "scale takes hours per eval; 0 means no cap.")
    parser.add_argument("--num-workers", type=int, default=None,
                         help="DataLoader worker processes for image loading/decoding. 0 = synchronous "
                              "(main process only) -- keep >0 to avoid GPU idling on CPU-bound decode.")
    parser.add_argument("--sequential-ar-steps", type=int, default=None,
                         help="Train the AR decoder in plain sequential mode (full teacher forcing, "
                              "unrestricted cross-attention) for this many initial steps before switching "
                              "to blockwise mode for the rest of the run -- see modules/decoder.py's "
                              "'Two-stage training' note. 0 (default) disables this and trains blockwise "
                              "from step 0.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-public", action="store_true",
                         help="Make the pushed HF checkpoint repo public (default: private).")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig(seed=args.seed)
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps
    if args.log_every:
        train_cfg.log_every = args.log_every
    if args.ckpt_every:
        train_cfg.ckpt_every = args.ckpt_every
    if args.sample_every is not None:
        train_cfg.sample_every = args.sample_every
    if args.max_eval_samples is not None:
        train_cfg.max_eval_samples = args.max_eval_samples
    if args.max_ar_eval_samples is not None:
        train_cfg.max_ar_eval_samples = args.max_ar_eval_samples
    if args.keep_unlearnable:
        train_cfg.filter_unlearnable = False
    if args.eval_every is not None:
        train_cfg.eval_every = args.eval_every
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.sequential_ar_steps is not None:
        train_cfg.sequential_ar_steps = args.sequential_ar_steps

    run_training(
        model_cfg, train_cfg, real_data_roots=args.real_data_dirs, dedup_manifest_path=args.dedup_manifest,
        tokenizer_dir=args.tokenizer_dir, checkpoint_root=args.checkpoint_root, run_name=args.run_name,
        push_to_hub=args.push_to_hub, hf_token=args.hf_token, hub_private=not args.hub_public,
        auto_batch_size=args.auto_batch_size, resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
