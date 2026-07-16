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


def build_combined_index(real_data_roots, source_repeat: dict = None) -> list:
    """Concatenates Sample lists across the given real_data output roots.
    `source_repeat` (keyed by the source id, i.e. the root's directory name)
    lets a smaller source's samples be duplicated N times -- simple
    oversampling via literal list repetition, no weighted-sampler infra."""
    source_repeat = source_repeat or {}
    samples = []
    for root in real_data_roots:
        root = Path(root)
        repeat = source_repeat.get(root.name, 1)
        samples.extend(load_real_data_manifest(root) * repeat)
    return samples
