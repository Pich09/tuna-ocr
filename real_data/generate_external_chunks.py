"""CLI: pull a small sample of real OCR line images from external Hugging
Face datasets and slice each into fixed-width overlapping chunks for
Conformer-encoder input windowing.

Usage:
    python -m real_data.generate_external_chunks --source mrrtmob --num-samples 100
    python -m real_data.generate_external_chunks --source all --num-samples 50
"""
import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_image_overlap
from .config import REAL_DATA_ROOT, EXTERNAL_DATASETS, ExternalChunkConfig
from .hf_datasets import ExternalDatasetError, load_external_samples


def process_source(source: str, out_dir: Path, num_samples: int, chunk_width: int, overlap: int, seed: int, shuffle_buffer: int, writer):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    n_chunks = 0
    for line_id, sample in enumerate(tqdm(
        load_external_samples(source, max_samples=num_samples, seed=seed, shuffle_buffer=shuffle_buffer),
        total=num_samples, desc=f"{source}",
    )):
        img = sample["image"].convert("RGB")
        text = sample["text"]

        full_fname = f"{source}_{line_id:06d}.jpg"
        img.save(images_dir / full_fname, quality=92)
        writer.writerow([full_fname, text, source, line_id, -1, 0, img.width, True])

        chunks = chunk_image_overlap(img, chunk_width=chunk_width, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_fname = f"{source}_{line_id:06d}_chunk{chunk_idx:02d}.jpg"
            chunk.image.save(images_dir / chunk_fname, quality=92)
            writer.writerow([chunk_fname, text, source, line_id, chunk_idx, chunk.x_offset, chunk.width, False])
            n_chunks += 1
        n_lines += 1

    return n_lines, n_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=list(EXTERNAL_DATASETS) + ["all"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--chunk-width", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=0,
                         help="Streaming shuffle buffer size (0 = no shuffle, natural stream order). "
                              "A buffer must fully fill, one network-fetched row at a time, before "
                              "anything is yielded, so only raise this on a fast connection.")
    args = parser.parse_args()

    chunk_cfg = ExternalChunkConfig()
    chunk_width = args.chunk_width or chunk_cfg.chunk_width
    overlap = args.overlap if args.overlap is not None else chunk_cfg.overlap

    sources = list(EXTERNAL_DATASETS) if args.source == "all" else [args.source]
    out_root = args.out_dir or (REAL_DATA_ROOT / "samples")

    summary = {}
    for source in sources:
        out_dir = out_root / source if args.source == "all" else out_root
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.tsv"

        try:
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["filename", "text", "source", "line_id", "chunk_idx", "x_offset", "width", "is_full_line"])
                n_lines, n_chunks = process_source(source, out_dir, args.num_samples, chunk_width, overlap, args.seed, args.shuffle_buffer, writer)
            summary[source] = f"{n_lines} lines, {n_chunks} chunks written"
        except ExternalDatasetError as e:
            summary[source] = f"FAILED ({e.reason})"
            if args.source != "all":
                print(f"{type(e).__name__}: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"warning: {e}")

    print()
    for source, result in summary.items():
        print(f"{source}: {result}")


if __name__ == "__main__":
    main()
