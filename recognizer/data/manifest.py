"""Reads real_data manifest.tsv files (the only manifest format this
pipeline consumes -- data_gen synthetic output is out of scope) into a flat
list of Sample records, ready for chunking + tokenization in dataset.py."""
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image_path: Path
    text: str
    source: str


def load_real_data_manifest(root: Path) -> list:
    """Reads root/manifest.tsv (columns: filename, text, source, line_id,
    chunk_idx, x_offset, width, is_full_line -- see
    real_data/generate_external_chunks.py) and keeps only is_full_line==True
    rows: the original whole-line image + full transcript. real_data's own
    pre-chunked rows (256px chunks, tuned for its standalone CLI) are not
    consumed here -- this pipeline re-chunks every whole-line image itself
    at its own 128px/16px setting (see recognizer/data/transforms.py)."""
    manifest_path = Path(root) / "manifest.tsv"
    images_dir = Path(root) / "images"
    samples = []
    with manifest_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["is_full_line"] != "True":
                continue
            samples.append(Sample(
                image_path=images_dir / row["filename"],
                text=row["text"],
                source=row["source"],
            ))
    return samples


def load_dedup_manifest(path: Path) -> list:
    """Reads the output of `real_data.deduplicate` (columns: image_path,
    text, source, line_id) -- a single manifest already pooled and
    deduplicated across whatever --real-data-dirs were fed into it, so no
    further per-root loading/concatenation is needed. `image_path` is
    already absolute (written that way by deduplicate.py)."""
    samples = []
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            samples.append(Sample(image_path=Path(row["image_path"]), text=row["text"], source=row["source"]))
    return samples


def build_combined_index(real_data_roots, source_repeat: dict = None) -> list:
    """Concatenates Sample lists across the given real_data output roots.
    `source_repeat` (keyed by the source id, i.e. the root's directory name)
    lets a smaller source's samples be duplicated N times -- simple
    oversampling via literal list repetition, no weighted-sampler infra.

    Prefer `load_dedup_manifest` (via `real_data.deduplicate`'s output) over
    this when training for real -- this function does NOT deduplicate
    across the given roots, so duplicate/near-duplicate images across
    sources (the exact problem `real_data.deduplicate` exists to catch)
    will still make it into the training set if used directly."""
    source_repeat = source_repeat or {}
    samples = []
    for root in real_data_roots:
        root = Path(root)
        repeat = source_repeat.get(root.name, 1)
        samples.extend(load_real_data_manifest(root) * repeat)
    return samples
