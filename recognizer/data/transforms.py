"""Image loading/normalization + chunking -- the encoder's literal input
unit is a fixed-width chunk, not the whole line (see recognizer/config.py's
chunk_width/chunk_overlap and the top-level plan's Step 0.5)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from real_data.chunking import chunk_image_overlap

BACKGROUND_VALUE = 0.98  # near-white in [0,1] after normalization below; matches
                          # data_gen's off-white line backgrounds, so padding
                          # isn't taught as "ink"


def load_and_normalize(image_path: Path, target_height: int = 64) -> torch.Tensor:
    """Grayscale, resize to a fixed height (keeping aspect ratio), tensor in
    [0,1]. No dataset-computed mean/std -- deliberately simple for v1."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    new_w = max(1, round(w * target_height / h))
    img = img.resize((new_w, target_height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def pad_to_width(chunk: torch.Tensor, target_width: int) -> torch.Tensor:
    """Right-pads a (1,H,W) tensor with W < target_width up to target_width
    using BACKGROUND_VALUE -- only ever needed for the rare whole-line
    image shorter than one chunk (chunk_image_overlap returns it unpadded)."""
    _, h, w = chunk.shape
    if w >= target_width:
        return chunk
    pad = torch.full((1, h, target_width - w), BACKGROUND_VALUE, dtype=chunk.dtype)
    return torch.cat([chunk, pad], dim=2)


def chunk_line_image(image_path: Path, chunk_width: int, chunk_overlap: int, target_height: int = 64):
    """Loads the whole-line image and returns (chunk_tensors, valid_widths):
    chunk_tensors is a list of (1, target_height, chunk_width) tensors (the
    rare narrower-than-one-chunk line is right-padded), valid_widths is the
    real (unpadded) pixel width of each chunk, for the encoder's frame-count
    accounting."""
    line_tensor = load_and_normalize(image_path, target_height)
    line_img = Image.fromarray((line_tensor.squeeze(0).numpy() * 255).astype(np.uint8), mode="L")
    chunks = chunk_image_overlap(line_img, chunk_width=chunk_width, overlap=chunk_overlap)

    chunk_tensors, valid_widths = [], []
    for c in chunks:
        arr = np.asarray(c.image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
        valid_widths.append(c.width)
        chunk_tensors.append(pad_to_width(t, chunk_width))
    return chunk_tensors, valid_widths
