"""Batch evaluation: Character Error Rate (CER), chosen over WER because
Khmer word segmentation is itself ambiguous (the entire reason
khmer_segmentation.py exists upstream) -- WER would partly measure
segmentation-heuristic agreement rather than recognition accuracy. Also
reports a CTC-only greedy decode CER as an independent sanity check.

Usage:
    python -m recognizer.evaluate --checkpoint recognizer/checkpoints/v1/last.pt \\
        --real-data-dirs real_data/samples/darayut_scene_text --tokenizer-dir recognizer/tokenizer/assets
"""
import argparse
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TOKENIZER_ASSETS_DIR
from .data.dataset import OCRLineDataset, make_collate_fn
from .data.manifest import build_combined_index
from .modules.model import Recognizer
from .tokenizer.khmer_ocr_tokenizer import KhmerOcrTokenizer


def compute_cer(refs: list, hyps: list) -> float:
    total_edits = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
    total_len = sum(len(r) for r in refs) or 1
    return total_edits / total_len


def ctc_greedy_decode(ctc_logits: torch.Tensor, ctc_blank_id: int) -> list:
    """argmax + collapse-repeats + drop-blank, per sample."""
    ids = ctc_logits.argmax(dim=-1)  # (B,T)
    out = []
    for row in ids.tolist():
        collapsed, prev = [], None
        for i in row:
            if i != prev and i != ctc_blank_id:
                collapsed.append(i)
            prev = i
        out.append(collapsed)
    return out


def load_model(checkpoint_path: Path, tokenizer: KhmerOcrTokenizer, device):
    state = torch.load(checkpoint_path, map_location=device)
    model_cfg = state["model_cfg"]
    model = Recognizer(model_cfg, tokenizer.vocab_size, bos_id=tokenizer.bos_id, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, model_cfg


@torch.no_grad()
def evaluate(checkpoint_path: Path, real_data_roots, tokenizer_dir=TOKENIZER_ASSETS_DIR, batch_size: int = 16,
             device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KhmerOcrTokenizer(tokenizer_dir)
    model, model_cfg = load_model(checkpoint_path, tokenizer, device)

    samples = build_combined_index(real_data_roots)
    ds = OCRLineDataset(samples, tokenizer, model_cfg)
    collate_fn = make_collate_fn(tokenizer, model_cfg.chunk_width)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    refs, ar_hyps, ctc_hyps = [], [], []
    ctc_blank_id = tokenizer.vocab_size
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        enc_out, enc_lengths, enc_block_ids = model.encode(batch["chunks"], batch["chunks_per_line"])
        max_blocks = int(batch["chunks_per_line"].max())
        ar_tokens = model.decoder.decode_greedy(enc_out, enc_lengths, enc_block_ids, max_blocks)
        ctc_logits = model.ctc_head(enc_out)
        ctc_tokens = ctc_greedy_decode(ctc_logits, ctc_blank_id)

        for text, ar_row, ctc_row in zip(batch["texts"], ar_tokens.tolist(), ctc_tokens):
            refs.append(text)
            ar_hyps.append(tokenizer.decode(ar_row))
            ctc_hyps.append(tokenizer.decode(ctc_row))

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
