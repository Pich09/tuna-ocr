"""Single-image inference CLI.

Usage:
    python -m recognizer.infer --checkpoint recognizer/checkpoints/v1/last.pt \\
        --image line.jpg --tokenizer-dir recognizer/tokenizer/assets
"""
import argparse
from pathlib import Path

import torch

from . import env_utils
from .config import TOKENIZER_ASSETS_DIR
from .data.transforms import chunk_line_image
from .evaluate import load_model
from .modules.decoder import truncate_blocks
from .tokenizer.khmer_ocr_tokenizer import KhmerOcrTokenizer


@torch.no_grad()
def infer(checkpoint_path: Path, image_path: Path, tokenizer_dir=TOKENIZER_ASSETS_DIR, keep_tags: bool = False,
           max_len: int = 256, device=None):
    device = device or env_utils.get_torch_device()
    tokenizer = KhmerOcrTokenizer(tokenizer_dir)
    model, model_cfg, _char_vocab = load_model(checkpoint_path, tokenizer, device)

    chunk_tensors, _ = chunk_line_image(image_path, model_cfg.chunk_width, model_cfg.chunk_overlap, model_cfg.img_height)
    chunk_batch = torch.stack(chunk_tensors).to(device)
    chunks_per_line = torch.tensor([len(chunk_tensors)], dtype=torch.long, device=device)

    enc_out, enc_lengths, enc_block_ids = model.encode(chunk_batch, chunks_per_line)
    max_blocks = min(len(chunk_tensors), max_len // max(1, model_cfg.max_tokens_per_block) + 1)
    tokens = model.decoder.decode_greedy(enc_out, enc_lengths, enc_block_ids, max_blocks)[0].tolist()
    tokens = truncate_blocks(tokens, model_cfg.max_tokens_per_block, tokenizer.eob_id, tokenizer.pad_id)
    return tokenizer.decode(tokens, strip_control=not keep_tags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_ASSETS_DIR)
    parser.add_argument("--keep-tags", action="store_true")
    args = parser.parse_args()
    print(infer(args.checkpoint, args.image, tokenizer_dir=args.tokenizer_dir, keep_tags=args.keep_tags))


if __name__ == "__main__":
    main()
