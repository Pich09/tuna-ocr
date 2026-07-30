"""Dataset/DataLoader over real_data output. Chunking (recognizer/config.py's
chunk_width/chunk_overlap) is the encoder's mandatory input unit, and chunk
boundaries double 1:1 as the blockwise-AR decoder's block boundaries -- see
the plan's Step 0.5/Step 3. Since no source provides real per-token
alignment, each sample's tagged token sequence is uniform-split into
`num_chunks` runs for teacher forcing (Step 3's documented approximation).
"""
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .transforms import chunk_line_image, open_image

# Per-source tally of block truncations, so the (common, expected) event is
# reported as one warning plus a running count instead of one warning per
# affected sample. The message text used to embed the offending token count,
# which made every occurrence textually unique -- defeating warnings' built-in
# dedup and emitting an unbounded stream into the notebook. At ~6% of samples
# that is one or two warnings per batch forever, which buries the training log
# and can take a Colab kernel down through sheer output volume.
_TRUNCATION_COUNTS = {}
_TRUNCATION_DROPPED = {}


def truncation_stats() -> dict:
    """{source: (num_blocks_truncated, num_tokens_dropped)} accumulated so far.
    Logged periodically by train.py -- this is real supervision being thrown
    away, so it belongs in the training log as a number, not as log spam."""
    return {s: (_TRUNCATION_COUNTS[s], _TRUNCATION_DROPPED.get(s, 0)) for s in _TRUNCATION_COUNTS}


def split_into_blocks(token_ids: list, num_blocks: int, max_tokens_per_block: int, eob_id: int, pad_id: int, source: str = "") -> list:
    """Uniform-interval split of `token_ids` into `num_blocks` contiguous
    runs, each formatted to exactly `max_tokens_per_block` ids: the real
    tokens, then `<eob>` + `<pad>`-fill if the run is shorter than the
    block, or truncated (with a logged warning) if longer."""
    num_blocks = max(1, num_blocks)
    n = len(token_ids)
    base, rem = divmod(n, num_blocks)
    rows, idx = [], 0
    for i in range(num_blocks):
        size = base + (1 if i < rem else 0)
        run = token_ids[idx: idx + size]
        idx += size
        if len(run) > max_tokens_per_block:
            first = source not in _TRUNCATION_COUNTS
            _TRUNCATION_COUNTS[source] = _TRUNCATION_COUNTS.get(source, 0) + 1
            _TRUNCATION_DROPPED[source] = (_TRUNCATION_DROPPED.get(source, 0)
                                           + len(run) - max_tokens_per_block)
            if first:
                # Once per source, with a stable message so warnings' own dedup
                # holds even if this is ever reached from several call sites.
                warnings.warn(
                    f"[{source}] AR block targets exceed max_tokens_per_block="
                    f"{max_tokens_per_block} and are being truncated (known v1 limitation "
                    f"of uniform block splitting -- see recognizer plan Step 3). Further "
                    f"occurrences are counted, not warned; see dataset.truncation_stats()."
                )
            run = run[:max_tokens_per_block]
        if len(run) < max_tokens_per_block:
            run = run + [eob_id] + [pad_id] * (max_tokens_per_block - len(run) - 1)
        rows.append(run)
    return rows


class OCRLineDataset(Dataset):
    def __init__(self, samples: list, tokenizer, cfg, char_vocab=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.cfg = cfg
        # When set, the CTC target is character ids instead of subword ids --
        # see data/char_vocab.py for the measurement that motivated this.
        # The AR target always stays subword.
        self.char_vocab = char_vocab

    def __len__(self):
        return len(self.samples)

    def image_width(self, idx: int) -> int:
        """Cheap width lookup (PIL lazy header read, no full pixel decode)
        used by BucketBatchSampler -- proxy for num_chunks without paying
        for the actual chunking pass."""
        with open_image(self.samples[idx].image_source) as img:
            return img.size[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chunk_tensors, valid_widths = chunk_line_image(
            sample.image_source, self.cfg.chunk_width, self.cfg.chunk_overlap, self.cfg.img_height,
        )
        token_ids = self.tokenizer.encode_plain(sample.text)
        ar_rows = split_into_blocks(
            token_ids, len(chunk_tensors), self.cfg.max_tokens_per_block,
            self.tokenizer.eob_id, self.tokenizer.pad_id, source=sample.source,
        )
        ctc_ids = self.char_vocab.encode(sample.text) if self.char_vocab else token_ids
        return {
            "chunks": chunk_tensors,
            "valid_widths": valid_widths,
            "ctc_target": torch.tensor(ctc_ids, dtype=torch.long),
            "ar_target": torch.tensor(ar_rows, dtype=torch.long),
            # Plain (unsegmented) token sequence + trailing <eos> -- used for the
            # sequential-AR training stage (see modules/decoder.py's
            # forward_sequential): unlike ar_target, this carries NO uniform-split
            # block-boundary artifacts, since sequential mode's cross-attention
            # isn't restricted to a per-block encoder-frame window and so has no
            # need for the (approximate) block-target partitioning at all.
            "ar_flat_target": torch.tensor(token_ids + [self.tokenizer.eos_id], dtype=torch.long),
            "text": sample.text,
        }


def make_collate_fn(tokenizer, chunk_width: int):
    pad_id = tokenizer.pad_id

    def collate_fn(batch: list) -> dict:
        chunks_per_line = torch.tensor([len(b["chunks"]) for b in batch], dtype=torch.long)
        all_chunks = torch.cat([torch.stack(b["chunks"]) for b in batch], dim=0)  # (N,1,H,chunk_width)
        valid_widths = torch.tensor([w for b in batch for w in b["valid_widths"]], dtype=torch.long)

        ctc_lengths = torch.tensor([len(b["ctc_target"]) for b in batch], dtype=torch.long)
        max_ctc_len = int(ctc_lengths.max())
        ctc_targets = torch.full((len(batch), max_ctc_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ctc_targets[i, : len(b["ctc_target"])] = b["ctc_target"]

        max_blocks = int(chunks_per_line.max())
        max_tokens_per_block = batch[0]["ar_target"].shape[1]
        ar_targets = torch.full((len(batch), max_blocks, max_tokens_per_block), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            nb = b["ar_target"].shape[0]
            ar_targets[i, :nb] = b["ar_target"]

        max_flat_len = max(len(b["ar_flat_target"]) for b in batch)
        ar_flat_targets = torch.full((len(batch), max_flat_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ar_flat_targets[i, : len(b["ar_flat_target"])] = b["ar_flat_target"]

        return {
            "chunks": all_chunks,
            "chunks_per_line": chunks_per_line,
            "valid_widths": valid_widths,
            "ctc_targets": ctc_targets,
            "ar_flat_targets": ar_flat_targets,
            "ctc_lengths": ctc_lengths,
            "ar_targets": ar_targets,
            "texts": [b["text"] for b in batch],
        }

    return collate_fn


# Batch entries that must stay on the host. `chunks_per_line` is never fed to a
# layer -- it only drives Python control flow (`.tolist()` in
# ConformerEncoder.forward, `.max()` in the decode paths). Moving it to the
# accelerator and reading it back forces a device->host sync on every step,
# which on XLA/TPU serializes the whole asynchronous execution pipeline (the
# graph must run to completion before the loop can even decide its shapes) and
# on CUDA costs a needless stall. Keeping it on the CPU makes those reads free.
HOST_ONLY_BATCH_KEYS = frozenset({"chunks_per_line"})


def move_batch(batch: dict, device) -> dict:
    """Moves a collated batch to `device`, leaving HOST_ONLY_BATCH_KEYS and
    non-tensor entries (e.g. `texts`) on the CPU."""
    return {
        k: (v.to(device) if torch.is_tensor(v) and k not in HOST_ONLY_BATCH_KEYS else v)
        for k, v in batch.items()
    }


def compute_widths(dataset: "OCRLineDataset", num_workers: int = 32, cache_path: Path = None) -> list:
    """Per-sample image width, threaded (I/O-bound: PIL's lazy header read
    releases the GIL during the actual file read) -- at millions of samples
    a serial scan is minutes of dead time. Shared by BucketBatchSampler (to
    bucket similar-width images together) and find_max_batch_size (to probe
    OOM safety against the actual widest -- i.e. worst-case memory -- batch,
    not an arbitrary one: since batches are width-bucketed, the batch of
    widest images is exactly the one most likely to OOM mid-training if the
    probe under-estimated it).

    Even threaded, this is one stat/header-read per sample and is bound by
    the filesystem, not the CPU -- ~25 min for a 4M-sample corpus on a busy
    disk, paid before the first training step on EVERY (re)start. The result
    is a pure function of the sample list, so `cache_path` memoizes it to
    disk: subsequent runs over the same data load it in seconds. The cache
    is keyed on the sample list's identity (length + first/last path), so a
    changed dataset misses the cache and recomputes rather than silently
    reusing stale widths."""
    def _sample_key(s):
        # image_path gives a stable identity for path-backed samples; bytes-backed
        # samples (Arrow-packed, see manifest.py) have no path, so fall back to a
        # cheap (not cryptographic) proxy -- byte length + text -- good enough to
        # invalidate the cache on a genuinely different dataset.
        if s.image_path is not None:
            return str(s.image_path)
        return f"bytes:{len(s.image_bytes)}:{s.text}"

    key = None
    if cache_path is not None and len(dataset) > 0:
        key = (f"{len(dataset)}|{_sample_key(dataset.samples[0])}|"
               f"{_sample_key(dataset.samples[-1])}")
        try:
            with open(cache_path, encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("key") == key and len(blob.get("widths", ())) == len(dataset):
                return blob["widths"]
        except (OSError, ValueError):
            pass  # missing/corrupt cache -> just recompute

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        widths = list(pool.map(dataset.image_width, range(len(dataset))))

    if key is not None:
        try:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_name(cache_path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"key": key, "widths": widths}, f)
            tmp.replace(cache_path)  # atomic: never leave a half-written cache
        except OSError:
            pass  # caching is an optimization; never fail training over it
    return widths


class BucketBatchSampler(Sampler):
    """Sorts indices once by (cheap-to-read) image width so each batch has
    similar chunk counts, minimizing pad waste, then shuffles BATCH order
    (not sample order) each epoch."""

    def __init__(self, dataset: OCRLineDataset, batch_size: int, shuffle: bool = True, generator=None,
                 widths: list = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        if widths is None:
            widths = compute_widths(dataset)
        indexed = sorted(enumerate(widths), key=lambda p: p[1])
        sorted_idx = [i for i, _ in indexed]
        self.batches = [sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)]

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            order = torch.randperm(len(order), generator=self.generator).tolist()
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)
