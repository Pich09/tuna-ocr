"""CLI: deduplicate pulled real_data line images across one or more
per-source .arrow files (see pack_arrow.py) -- exact + near-duplicate image
detection, see dedup.py -- writing a single combined Arrow file of
survivors plus a JSON report. Run this after pack_arrow.py and before
recognizer.train -- point recognizer.train's --dedup-manifest at this
script's output instead of listing raw source files, so training never
sees duplicate images (including, e.g., a source that turns out to mirror
another source's rows).

Reads/writes Arrow, not loose image files: keeps the pipeline free of the
millions-of-tiny-files problem pack_arrow.py exists to solve -- survivors'
image bytes are copied from the input .arrow files straight into the
output .arrow file, never touching disk as individual images.

Usage:
    python -m real_data.deduplicate --arrow-files real_data/samples/deepcopy_khmer_text_recognition.arrow \\
        real_data/samples/chanrith_ocr_image_line.arrow real_data/samples/darayut_scene_text.arrow \\
        real_data/samples/sokheng_synthetic_v1.arrow \\
        --out real_data/samples/dedup.arrow \\
        --near-dup-threshold 0
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm

from .config import REAL_DATA_ROOT
from .dedup import Record, find_duplicates
from .pack_arrow import SCHEMA


def load_arrow_records(path: Path) -> list:
    with pa.memory_map(str(path), "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    texts = table.column("text").to_pylist()
    sources = table.column("source").to_pylist()
    line_ids = table.column("line_id").to_pylist()
    images = table.column("image").to_pylist()
    return [
        Record(image_bytes=img, text=t, source=s, line_id=str(lid))
        for t, s, lid, img in zip(texts, sources, line_ids, images)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-files", nargs="+", required=True,
                         help="One or more per-source .arrow files (see pack_arrow.py) to pool and "
                              "deduplicate.")
    parser.add_argument("--out", type=Path, default=REAL_DATA_ROOT / "samples" / "dedup.arrow")
    parser.add_argument("--near-dup-threshold", type=int, default=10,
                         help="Perceptual-hash (hash_size=16) Hamming distance <= this counts as a "
                              "near-duplicate -- see dedup.py's find_duplicates docstring for how this "
                              "default was empirically tuned. 0 disables near-duplicate detection "
                              "(exact-hash dedup only) -- near-duplicate detection is brute-force "
                              "O(n^2); pass 0 once the pooled record count reaches the tens of "
                              "thousands+, where the pairwise pass becomes impractically slow.")
    args = parser.parse_args()

    records = []
    for f in args.arrow_files:
        source_records = load_arrow_records(Path(f))
        print(f"{f}: {len(source_records)} whole-line samples")
        records.extend(source_records)

    if not records:
        raise SystemExit("No records found in the given --arrow-files.")

    print(f"hashing {len(records)} images (exact + near-duplicate detection)...")
    result = find_duplicates(
        list(tqdm(records, desc="records")), near_dup_threshold=args.near_dup_threshold,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "text": [r.text for r in result.kept],
        "source": [r.source for r in result.kept],
        "line_id": [int(r.line_id) for r in result.kept],
        "image": [r.image_bytes for r in result.kept],
    }, schema=SCHEMA)
    with pa.OSFile(str(args.out), "wb") as sink:
        with pa.ipc.new_file(sink, SCHEMA) as writer:
            writer.write_table(table)

    per_source_kept = {}
    for r in result.kept:
        per_source_kept[r.source] = per_source_kept.get(r.source, 0) + 1

    report = {
        "input_records": len(records),
        "kept": len(result.kept),
        "exact_duplicates_removed": result.exact_duplicate_count,
        "near_duplicates_removed": result.near_duplicate_count,
        "near_dup_threshold": args.near_dup_threshold,
        "duplicate_text_count_informational": result.duplicate_text_count,
        "per_source_kept": per_source_kept,
        "exact_duplicate_example_groups": [
            [{"source": r.source, "line_id": r.line_id, "text": r.text} for r in group]
            for group in result.exact_duplicate_groups[:20]
        ],
        "near_duplicate_example_groups": [
            [{"source": r.source, "line_id": r.line_id, "text": r.text} for r in group]
            for group in result.near_duplicate_groups[:20]
        ],
    }
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\nkept {len(result.kept)}/{len(records)} "
          f"(-{result.exact_duplicate_count} exact, -{result.near_duplicate_count} near-duplicate)")
    print(f"arrow file: {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")
    print(f"report:     {report_path}")
    if result.duplicate_text_count:
        print(f"note: {result.duplicate_text_count} kept samples share a transcript with another "
              f"kept sample (different image, not removed -- see dedup.py docstring).")


if __name__ == "__main__":
    main()
