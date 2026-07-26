"""Duplicate detection for pulled real_data line images -- pure functions,
no CLI/manifest-format knowledge here (see deduplicate.py for the CLI that
reads/writes Arrow files). Operates on whole-line images (the training unit
recognizer/ actually consumes, per its is_full_line-only manifest loader),
not the pre-cut chunk crops.

Two kinds of duplication are detected, and only one is auto-removed:
- **Exact image duplicates** (byte-identical files, e.g. the same line
  photographed/rendered twice, or the same row appearing under two source
  names -- this was the concern that motivated this pipeline, see
  real_data/README.md): detected via SHA-256 of the raw image bytes, cheap
  and unambiguous. Always removed (kept: first occurrence, by input order).
- **Near-duplicate images** (perceptually near-identical but not
  byte-identical -- e.g. a re-encoded/re-compressed copy): detected via
  perceptual hashing (`imagehash.phash`) with a Hamming-distance threshold.
  Removed by default (`near_dup_threshold` > 0); pass 0 to disable and only
  do exact dedup.

Same-transcript-different-image rows are NOT deduplicated by this module --
many legitimately distinct images can share a transcript (common Khmer
words rendered by different writers/fonts), so that's a diversity signal
worth reporting, not a redundancy worth deleting. See DedupResult.
duplicate_text_count.

Near-duplicate detection is brute-force O(n^2) pairwise Hamming distance
over survivors of the exact-dedup pass -- fine at hundreds to a few
thousand samples per source, NOT at tens/hundreds of thousands+ (a 338k-row
pull is ~5.7*10^10 pairwise comparisons) -- pass --near-dup-threshold 0 at
that scale to skip this pass entirely (exact-hash dedup is still O(n) and
stays cheap regardless of scale).
"""
import hashlib
import io
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class Record:
    image_bytes: bytes
    text: str
    source: str
    line_id: str


@dataclass
class DedupResult:
    kept: list          # list[Record], survivors in input order
    exact_duplicate_count: int
    near_duplicate_count: int
    duplicate_text_count: int  # informational only, not filtered -- see module docstring
    exact_duplicate_groups: list = field(default_factory=list)   # list[list[Record]], group[0] kept
    near_duplicate_groups: list = field(default_factory=list)    # list[list[Record]], group[0] kept


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phash_bytes(data: bytes, hash_size: int = 16):
    import imagehash

    with Image.open(io.BytesIO(data)) as img:
        return imagehash.phash(img, hash_size=hash_size)


def find_duplicates(records: list, near_dup_threshold: int = 10, hash_size: int = 16) -> DedupResult:
    """records: list[Record], already in the priority order ties should
    resolve by (earlier record in the list wins and is kept).

    hash_size/near_dup_threshold defaults were tuned empirically, not
    guessed: OCR line crops are mostly-uniform background + thin text, so
    the DEFAULT imagehash.phash hash_size=8 (a 64-bit hash / 8x8 DCT) is too
    coarse for them -- two genuinely different line images (different text
    entirely) measured Hamming distance 4 at hash_size=8, exactly at the
    threshold that was originally used here, causing a real false-positive
    removal (verified against real_data/samples/chanrith_ocr_image_line
    while building this pipeline). At hash_size=16 (256-bit hash) the same
    pair separates to distance 114, while a genuine near-duplicate (the same
    image re-JPEG-compressed at quality=50) measures distance 2 -- a huge,
    clean margin. near_dup_threshold=10 sits safely inside that margin."""
    # -- pass 1: exact byte-identical duplicates --
    by_sha = {}
    for r in records:
        by_sha.setdefault(sha256_bytes(r.image_bytes), []).append(r)

    survivors, exact_groups, exact_dup_count = [], [], 0
    for group in by_sha.values():
        survivors.append(group[0])
        if len(group) > 1:
            exact_groups.append(group)
            exact_dup_count += len(group) - 1

    # -- pass 2: near-duplicate perceptual-hash clustering --
    near_groups, near_dup_count = [], 0
    kept = survivors
    if near_dup_threshold > 0 and len(survivors) > 1:
        hashes = [(r, phash_bytes(r.image_bytes, hash_size=hash_size)) for r in survivors]
        consumed = [False] * len(hashes)
        kept = []
        for i, (r_i, h_i) in enumerate(hashes):
            if consumed[i]:
                continue
            group = [r_i]
            for j in range(i + 1, len(hashes)):
                if consumed[j]:
                    continue
                r_j, h_j = hashes[j]
                if h_i - h_j <= near_dup_threshold:
                    group.append(r_j)
                    consumed[j] = True
            kept.append(r_i)
            if len(group) > 1:
                near_groups.append(group)
                near_dup_count += len(group) - 1

    # -- informational: same-transcript-different-image count (not filtered) --
    by_text = {}
    for r in kept:
        by_text.setdefault(r.text, []).append(r)
    duplicate_text_count = sum(len(v) - 1 for v in by_text.values() if len(v) > 1)

    return DedupResult(
        kept=kept,
        exact_duplicate_count=exact_dup_count,
        near_duplicate_count=near_dup_count,
        duplicate_text_count=duplicate_text_count,
        exact_duplicate_groups=exact_groups,
        near_duplicate_groups=near_groups,
    )
