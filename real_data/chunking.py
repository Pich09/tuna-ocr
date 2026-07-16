"""Fixed-width overlapping horizontal chunking for encoder input windowing.

Source-agnostic: operates on any PIL.Image line crop (synthetic or from an
external dataset). This does NOT re-derive per-chunk text labels -- a
chunked line keeps a single whole-line transcript as its target. Chunking
only bounds each block's pixel width for a Conformer-style encoder that
expects fixed-width input blocks (see data_gen/README.md's recognizer
target format / blockwise-AR decoder discussion).

chunk_width/overlap defaults in config.ExternalChunkConfig are placeholders
until the actual Conformer's block size is chosen -- that choice is out of
scope for this module.
"""
from dataclasses import dataclass

from PIL import Image


@dataclass
class Chunk:
    image: Image.Image
    x_offset: int  # left edge in the ORIGINAL image's coordinate space
    width: int      # == image.width, kept for convenient manifest writing


def chunk_image_overlap(img: Image.Image, chunk_width: int, overlap: int) -> list[Chunk]:
    """Slices `img` left-to-right into `chunk_width`-wide chunks.

    Chunk 0 starts at x=0. Chunk i (i >= 1) starts at
    `i * (chunk_width - overlap)`, so it overlaps backward into the
    previous chunk's tail by `overlap` px -- this is what keeps a Khmer
    grapheme cluster that straddled the previous chunk's right edge whole
    in the next chunk too.

    If `img` is narrower than `chunk_width`, it's returned as a single
    unpadded chunk (never invent blank content for a real line already
    shorter than one block). Otherwise every chunk is exactly
    `chunk_width` wide; the last chunk is shifted left (not blank-padded)
    so it doesn't run past the image and every chunk has only real pixel
    content.
    """
    if not (0 <= overlap < chunk_width):
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_width, got overlap={overlap}, chunk_width={chunk_width}")

    w, h = img.size
    if w <= chunk_width:
        return [Chunk(image=img, x_offset=0, width=w)]

    stride = chunk_width - overlap
    chunks = []
    x = 0
    while True:
        start = min(x, w - chunk_width)
        chunks.append(Chunk(image=img.crop((start, 0, start + chunk_width, h)), x_offset=start, width=chunk_width))
        if start == w - chunk_width:
            break
        x += stride
    return chunks
