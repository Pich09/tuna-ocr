"""Reads sample lists into a flat list of Sample records, ready for
chunking + tokenization in dataset.py. Two source formats are supported:

- real_data manifest.tsv + loose image files (load_real_data_manifest) --
  the original format, still used by the small-scale
  notebooks/train_diagnose_eval.ipynb loop where the file count is small
  enough not to matter.
- A single packed Arrow file (load_dedup_arrow, see real_data/pack_arrow.py
  and real_data/deduplicate.py) -- used at real training scale, where
  millions of loose per-image files becomes a genuine filesystem problem.
  Image bytes are read once, up front, into each Sample -- the whole
  dataset's image bytes doing this at once is a few GB at the scales this
  pipeline pulls, well within memory, unlike a persisted-tensor cache which
  would be several times larger (raw pixels vs. JPEG-compressed bytes).

Sample carries exactly one of image_path / image_bytes -- see
recognizer/data/transforms.py for the code that accepts either.
"""
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    text: str
    source: str
    image_path: Path = None
    image_bytes: bytes = None

    @property
    def image_source(self):
        """Whichever of image_path/image_bytes is set -- what
        transforms.chunk_line_image actually consumes."""
        return self.image_bytes if self.image_bytes is not None else self.image_path


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


def load_dedup_arrow(path: Path) -> list:
    """Reads a packed+deduplicated Arrow file (real_data.deduplicate's
    output, or real_data.pack_arrow's output directly if dedup was
    skipped) into Samples backed by in-memory image bytes rather than a
    file path."""
    import pyarrow as pa

    with pa.memory_map(str(path), "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    texts = table.column("text").to_pylist()
    sources = table.column("source").to_pylist()
    images = table.column("image").to_pylist()
    return [
        Sample(image_bytes=img, text=t, source=s)
        for t, s, img in zip(texts, sources, images)
    ]


def load_dedup_manifest(path: Path) -> list:
    """Reads the output of `real_data.deduplicate`. Dispatches on file
    extension: `.arrow` (current format, see load_dedup_arrow) or `.tsv`
    (older path-based manifest format, columns: image_path, text, source,
    line_id -- `image_path` already absolute)."""
    path = Path(path)
    if path.suffix == ".arrow":
        return load_dedup_arrow(path)
    samples = []
    with path.open(encoding="utf-8") as f:
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
