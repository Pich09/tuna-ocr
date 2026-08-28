"""Batch evaluation: Character Error Rate (CER), chosen over WER because
Khmer word segmentation is itself ambiguous (the entire reason
khmer_segmentation.py exists upstream) -- WER would partly measure
segmentation-heuristic agreement rather than recognition accuracy. Also
reports a CTC-only greedy decode CER as an independent sanity check.

Usage:
    python -m recognizer.evaluate --checkpoint recognizer/checkpoints/v1/latest.pt \\
        --real-data-dirs real_data/samples/darayut_scene_text --tokenizer-dir recognizer/tokenizer/assets
"""
import argparse
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader

from . import env_utils
from .config import ModelConfig, TOKENIZER_ASSETS_DIR
from .data.char_vocab import CharVocab
from .data.dataset import OCRLineDataset, make_collate_fn, move_batch
from .data.manifest import build_combined_index
from .modules.decoder import truncate_blocks
from .modules.model import Recognizer
from .tokenizer.khmer_ocr_tokenizer import KhmerOcrTokenizer


def compute_cer(refs: list, hyps: list) -> float:
    total_edits = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
    total_len = sum(len(r) for r in refs) or 1
    return total_edits / total_len


def ctc_greedy_decode(ctc_logits: torch.Tensor, ctc_blank_id: int, enc_lengths: torch.Tensor = None) -> list:
    """argmax + collapse-repeats + drop-blank, per sample. `enc_lengths`
    bounds each row to its real frames: enc_out is zero-padded to the batch
    max, and the CTC head's argmax over those zero vectors is an arbitrary
    class -- without the bound, every shorter line in a batch grew a garbage
    tail decoded from padding."""
    ids = ctc_logits.argmax(dim=-1).cpu()  # (B,T); one transfer, not one .item() per frame
    lengths = enc_lengths.cpu().tolist() if enc_lengths is not None else [ids.shape[1]] * ids.shape[0]
    out = []
    for row, n in zip(ids.tolist(), lengths):
        collapsed, prev = [], None
        for i in row[:n]:
            if i != prev and i != ctc_blank_id:
                collapsed.append(i)
            prev = i
        out.append(collapsed)
    return out


def load_model(checkpoint_path: Path, tokenizer: KhmerOcrTokenizer, device):
    # weights_only=False: self-produced checkpoint (stores our own ModelConfig
    # dataclass alongside the state dicts) -- see train.py's resume path for
    # the same note on PyTorch 2.6+'s weights_only=True default.
    # map_location="cpu" always: torch.load's map_location doesn't reliably
    # understand XLA devices, so load to CPU and let model.to(device) below
    # move it (works identically for cuda/cpu/xla).
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    # Checkpoints from the character-CTC layout carry their own CTC label set;
    # older subword-CTC checkpoints have none and fall back to vocab_size + 1.
    char_vocab = CharVocab.from_json(state["char_vocab"]) if "char_vocab" in state else None
    model = Recognizer(model_cfg, tokenizer.vocab_size, bos_id=tokenizer.bos_id,
                       pad_id=tokenizer.pad_id,
                       ctc_vocab_size=char_vocab.size if char_vocab else None,
                       eos_id=tokenizer.eos_id, eob_id=tokenizer.eob_id).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, model_cfg, char_vocab


@torch.no_grad()
def evaluate(checkpoint_path: Path, real_data_roots, tokenizer_dir=TOKENIZER_ASSETS_DIR, batch_size: int = 16,
             device=None):
    device = device or env_utils.get_torch_device()
    tokenizer = KhmerOcrTokenizer(tokenizer_dir)
    model, model_cfg, char_vocab = load_model(checkpoint_path, tokenizer, device)

    samples = build_combined_index(real_data_roots)
    ds = OCRLineDataset(samples, tokenizer, model_cfg, char_vocab=char_vocab)
    collate_fn = make_collate_fn(tokenizer, model_cfg.chunk_width)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    refs, ar_hyps, ctc_hyps = [], [], []
    ctc_blank_id = char_vocab.blank_id if char_vocab else tokenizer.vocab_size
    for batch in loader:
        batch = move_batch(batch, device)
        enc_out, enc_lengths, enc_block_ids = model.encode(batch["chunks"], batch["chunks_per_line"])
        max_blocks = int(batch["chunks_per_line"].max())
        ar_tokens = model.decoder.decode_greedy(enc_out, enc_lengths, enc_block_ids, max_blocks)
        ctc_logits = model.ctc_head(enc_out)
        ctc_tokens = ctc_greedy_decode(ctc_logits, ctc_blank_id, enc_lengths=enc_lengths)

        chunks_per_line = batch["chunks_per_line"].tolist()
        for text, ar_row, ctc_row, n_blocks in zip(batch["texts"], ar_tokens.tolist(), ctc_tokens, chunks_per_line):
            refs.append(text)
            # Per-block truncation: within each K-slot block keep tokens up to
            # the first <eob>/<pad>, and drop blocks decoded past this line's
            # real chunk count (the batch decodes every line to the batch max).
            ar_row = truncate_blocks(ar_row, model_cfg.max_tokens_per_block,
                                     tokenizer.eob_id, tokenizer.pad_id, num_blocks=n_blocks)
            ar_hyps.append(tokenizer.decode(ar_row))
            # CTC labels are characters when a char vocab is present -- decoding
            # them through the subword tokenizer would be meaningless.
            ctc_hyps.append(char_vocab.decode(ctc_row) if char_vocab else tokenizer.decode(ctc_row))

    ar_cer = compute_cer(refs, ar_hyps)
    ctc_cer = compute_cer(refs, ctc_hyps)
    print(f"AR-decoder CER:  {ar_cer:.4f} ({len(refs)} samples)")
    print(f"CTC-greedy CER:  {ctc_cer:.4f} (independent sanity check)")
    return ar_cer, ctc_cer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-data-dirs", nargs="+", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_ASSETS_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.real_data_dirs, tokenizer_dir=args.tokenizer_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
