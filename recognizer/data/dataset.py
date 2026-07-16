"""Dataset/DataLoader over real_data output. Chunking (recognizer/config.py's
chunk_width/chunk_overlap) is the encoder's mandatory input unit, and chunk
boundaries double 1:1 as the blockwise-AR decoder's block boundaries -- see
the plan's Step 0.5/Step 3. Since no source provides real per-token
alignment, each sample's tagged token sequence is uniform-split into
`num_chunks` runs for teacher forcing (Step 3's documented approximation).
"""
import warnings

import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image

from .transforms import chunk_line_image


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
            warnings.warn(
                f"[{source}] block {i} has {len(run)} tokens > max_tokens_per_block="
                f"{max_tokens_per_block}; truncating (known v1 limitation of uniform "
                f"block splitting -- see recognizer plan Step 3)."
            )
            run = run[:max_tokens_per_block]
        if len(run) < max_tokens_per_block:
            run = run + [eob_id] + [pad_id] * (max_tokens_per_block - len(run) - 1)
        rows.append(run)
    return rows


class OCRLineDataset(Dataset):
    def __init__(self, samples: list, tokenizer, cfg):
        self.samples = samples
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.samples)

    def image_width(self, idx: int) -> int:
        """Cheap width lookup (PIL lazy header read, no full pixel decode)
        used by BucketBatchSampler -- proxy for num_chunks without paying
        for the actual chunking pass."""
        with Image.open(self.samples[idx].image_path) as img:
            return img.size[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chunk_tensors, valid_widths = chunk_line_image(
            sample.image_path, self.cfg.chunk_width, self.cfg.chunk_overlap, self.cfg.img_height,
        )
        token_ids = self.tokenizer.encode_plain(sample.text)
        ar_rows = split_into_blocks(
            token_ids, len(chunk_tensors), self.cfg.max_tokens_per_block,
            self.tokenizer.eob_id, self.tokenizer.pad_id, source=sample.source,
        )
        return {
            "chunks": chunk_tensors,
            "valid_widths": valid_widths,
            "ctc_target": torch.tensor(token_ids, dtype=torch.long),
            "ar_target": torch.tensor(ar_rows, dtype=torch.long),
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

        return {
            "chunks": all_chunks,
            "chunks_per_line": chunks_per_line,
            "valid_widths": valid_widths,
            "ctc_targets": ctc_targets,
            "ctc_lengths": ctc_lengths,
            "ar_targets": ar_targets,
            "texts": [b["text"] for b in batch],
        }

    return collate_fn


class BucketBatchSampler(Sampler):
    """Sorts indices once by (cheap-to-read) image width so each batch has
    similar chunk counts, minimizing pad waste, then shuffles BATCH order
    (not sample order) each epoch."""

    def __init__(self, dataset: OCRLineDataset, batch_size: int, shuffle: bool = True, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        widths = [(i, dataset.image_width(i)) for i in range(len(dataset))]
        widths.sort(key=lambda p: p[1])
        sorted_idx = [i for i, _ in widths]
        self.batches = [sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)]

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            order = torch.randperm(len(order), generator=self.generator).tolist()
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)
