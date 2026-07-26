"""CLI: pull a small sample of real OCR line images from external Hugging
Face datasets and slice each into fixed-width overlapping chunks for
Conformer-encoder input windowing.

Usage:
    python -m real_data.generate_external_chunks --source mrrtmob --num-samples 100
    python -m real_data.generate_external_chunks --source all --num-samples 50

Resuming: reruns for the same --out-dir auto-resume from the last complete
line rather than restarting the whole stream, since the source datasets are
streamed (no random access) and re-fetching everything from row 0 on every
crash/restart is wasteful for multi-million-row pulls. This only works with
the default --shuffle-buffer 0 (natural stream order) -- resuming a shuffled
stream would land on a different sample order. Pass --restart to force
starting over from scratch.
"""
import argparse
import csv
import errno
import itertools
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_image_overlap
from .config import REAL_DATA_ROOT, EXTERNAL_DATASETS, ExternalChunkConfig
from .hf_datasets import ExternalDatasetError, load_external_samples

# Lines per images/ subdirectory. A single flat directory can hit ext4's
# htree entry limit around ~10M files (fails with ENOSPC on specific
# filenames regardless of actual free space) well before a 12M-row pull
# like chanrith_ocr_image_line finishes -- ~4 files/line here keeps each
# shard well under that.
SHARD_SIZE = 5000


def _shard_name(line_id: int) -> str:
    return f"shard_{line_id // SHARD_SIZE:05d}"


def _resume_point(manifest_path: Path, images_dir: Path, source: str) -> int:
    """Returns the line_id to resume writing from, having trimmed the
    manifest and any image files at/after that point. The last line found
    is always dropped and redone, since a crash may have interrupted it
    mid-write (full-line row written but not all its chunks, or vice versa).
    """
    if not manifest_path.exists():
        return 0

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if len(rows) <= 1:
        return 0
    header, body = rows[0], rows[1:]

    line_ids = {int(row[3]) for row in body}
    resume_from = max(line_ids)  # redo the last (possibly partial) line too

    kept = [row for row in body if int(row[3]) < resume_from]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(kept)

    if images_dir.exists():
        for path in images_dir.rglob(f"{source}_*.jpg"):
            # filenames are "{source}_{line_id:06d}[...].jpg", optionally
            # under a "shard_NNNNN/" subdirectory.
            digits = path.name[len(source) + 1: len(source) + 7]
            if digits.isdigit() and int(digits) >= resume_from:
                path.unlink()

    return resume_from


def _save_with_retry(img, path, retries: int = 6, initial_delay: float = 10.0):
    """Saves an image, retrying on ENOSPC with exponential backoff.

    This machine is shared with other jobs (e.g. notebooks in unrelated
    projects) whose own disk usage can transiently fill the disk to zero
    free bytes and then release it moments later -- a real crash-worthy
    OSError in isolation, but usually just noise here. Anything else
    (permission errors, etc.) still raises immediately.
    """
    delay = initial_delay
    for attempt in range(retries + 1):
        try:
            img.save(path, quality=92)
            return
        except OSError as e:
            if e.errno != errno.ENOSPC or attempt == retries:
                raise
            print(f"\nENOSPC saving {path}, retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{retries})...", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 300)


def process_source(source: str, out_dir: Path, num_samples: int, chunk_width: int, overlap: int, seed: int, shuffle_buffer: int, writer, resume_from: int = 0):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = load_external_samples(source, max_samples=num_samples, seed=seed, shuffle_buffer=shuffle_buffer)
    if resume_from:
        samples = itertools.islice(samples, resume_from, None)

    n_lines = 0
    n_chunks = 0
    for line_id, sample in enumerate(tqdm(
        samples, total=num_samples, initial=resume_from, desc=f"{source}",
    ), start=resume_from):
        img = sample["image"].convert("RGB")
        text = sample["text"]

        shard = _shard_name(line_id)
        shard_dir = images_dir / shard
        shard_dir.mkdir(exist_ok=True)

        full_fname = f"{shard}/{source}_{line_id:06d}.jpg"
        _save_with_retry(img, images_dir / full_fname)
        writer.writerow([full_fname, text, source, line_id, -1, 0, img.width, True])

        chunks = chunk_image_overlap(img, chunk_width=chunk_width, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_fname = f"{shard}/{source}_{line_id:06d}_chunk{chunk_idx:02d}.jpg"
            _save_with_retry(chunk.image, images_dir / chunk_fname)
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
    parser.add_argument("--restart", action="store_true",
                         help="Ignore any existing manifest/images for this out-dir and start over "
                              "from row 0, instead of the default auto-resume behavior.")
    args = parser.parse_args()

    chunk_cfg = ExternalChunkConfig()
    chunk_width = args.chunk_width or chunk_cfg.chunk_width
    overlap = args.overlap if args.overlap is not None else chunk_cfg.overlap

    sources = list(EXTERNAL_DATASETS) if args.source == "all" else [args.source]
    default_root = REAL_DATA_ROOT / "samples"

    summary = {}
    for source in sources:
        # Default (no --out-dir given): always nest under samples/<source>/, whether
        # this is a single-source or --source all call -- a single named source with
        # no explicit --out-dir used to fall through to the flat samples/ root
        # instead, so every source clobbered the SAME manifest.tsv (each one's
        # resume logic then found the *previous* source's leftover rows and thought
        # it was resuming its own data). --out-dir, when explicitly passed for a
        # single named source, is still honored exactly as given (e.g. to point one
        # source at a custom location) -- it's only the *default* that was wrong.
        if args.out_dir is not None and args.source != "all":
            out_dir = args.out_dir
        else:
            out_dir = default_root / source
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.tsv"
        images_dir = out_dir / "images"

        if args.restart and manifest_path.exists():
            manifest_path.unlink()
            resume_from = 0
        else:
            resume_from = _resume_point(manifest_path, images_dir, source)

        if resume_from:
            print(f"{source}: resuming from line {resume_from} (found existing manifest)")

        try:
            mode = "a" if resume_from else "w"
            with manifest_path.open(mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                if not resume_from:
                    writer.writerow(["filename", "text", "source", "line_id", "chunk_idx", "x_offset", "width", "is_full_line"])
                n_lines, n_chunks = process_source(source, out_dir, args.num_samples, chunk_width, overlap, args.seed, args.shuffle_buffer, writer, resume_from)
            summary[source] = f"{n_lines} lines, {n_chunks} chunks written (resumed from {resume_from})" if resume_from else f"{n_lines} lines, {n_chunks} chunks written"
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
    # os._exit() rather than letting main() return into normal interpreter
    # shutdown: the `datasets` library's streaming path spawns background
    # fsspec/aiohttp threads that aren't always joined before Python starts
    # finalizing, which can crash with "Fatal Python error: PyGILState_Release
    # ... thread state must be current when releasing" -- observed on Colab,
    # *after* all real work (writing the manifest/images) had already
    # completed successfully. os._exit() skips that finalization path
    # entirely, so a caller checking the exit code sees the real outcome
    # (0 = actually succeeded) instead of a spurious crash on the way out.
    import os

    try:
        main()
        exit_code = 0
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
    os._exit(exit_code)
