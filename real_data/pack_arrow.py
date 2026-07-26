"""CLI: pack a single pulled source's whole-line images into one Arrow IPC
file, so the (potentially hundreds of thousands of) loose per-image .jpg
files under real_data/samples/<source>/images/ can be deleted afterward --
millions of tiny files is real pain at pull scale (filesystem overhead,
slow upload/listing on Kaggle/Colab), and Arrow gives one file per source
with zero-copy memory-mapped read access at training time instead.

Image bytes are stored exactly as pulled -- no re-encoding, no resize, no
quality/format change -- only the *container* changes (one Arrow file
instead of N loose files + a manifest.tsv). Only is_full_line==True rows
are packed: that's the only row type recognizer/ actually trains on (see
recognizer/data/manifest.py), so the pre-cut chunk-crop rows a manifest.tsv
may also contain are intentionally skipped.

Usage:
    python -m real_data.pack_arrow --source chanrith_ocr_image_line
    python -m real_data.pack_arrow --source chanrith_ocr_image_line --delete-raw
"""
import argparse
import csv
import shutil
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm

from .config import REAL_DATA_ROOT

SCHEMA = pa.schema([
    ("text", pa.string()),
    ("source", pa.string()),
    ("line_id", pa.int64()),
    ("image", pa.binary()),
])


def pack_source(source_dir: Path, out_path: Path) -> int:
    manifest_path = source_dir / "manifest.tsv"
    images_dir = source_dir / "images"

    texts, sources, line_ids, images = [], [], [], []
    with manifest_path.open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f, delimiter="\t") if row["is_full_line"] == "True"]

    for row in tqdm(rows, desc=f"packing {source_dir.name}"):
        image_bytes = (images_dir / row["filename"]).read_bytes()
        texts.append(row["text"])
        sources.append(row["source"])
        line_ids.append(int(row["line_id"]))
        images.append(image_bytes)

    table = pa.table({"text": texts, "source": sources, "line_id": line_ids, "image": images}, schema=SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(out_path), "wb") as sink:
        with pa.ipc.new_file(sink, SCHEMA) as writer:
            writer.write_table(table)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                         help="Source directory name under real_data/samples/ (e.g. as pulled by "
                              "generate_external_chunks.py).")
    parser.add_argument("--samples-root", type=Path, default=REAL_DATA_ROOT / "samples")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output .arrow path (default: <samples-root>/<source>.arrow)")
    parser.add_argument("--delete-raw", action="store_true",
                         help="Delete the source's raw images/ directory + manifest.tsv after a "
                              "successful pack -- only pass this once you've confirmed the .arrow "
                              "file is what you want to keep; this is not reversible.")
    args = parser.parse_args()

    source_dir = args.samples_root / args.source
    out_path = args.out or (args.samples_root / f"{args.source}.arrow")

    n = pack_source(source_dir, out_path)
    print(f"packed {n} whole-line samples from {source_dir} -> {out_path} "
          f"({out_path.stat().st_size / 1e9:.2f} GB)")

    if args.delete_raw:
        shutil.rmtree(source_dir)
        print(f"deleted raw source directory: {source_dir}")


if __name__ == "__main__":
    main()
