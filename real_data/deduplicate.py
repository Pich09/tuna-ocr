"""CLI: deduplicate pulled real_data line images across one or more source
directories (exact + near-duplicate image detection, see dedup.py), writing
a single combined manifest of survivors plus a JSON report. Run this after
generate_external_chunks.py and before recognizer.train -- point
recognizer.train's --dedup-manifest at this script's output instead of
listing raw --real-data-dirs, so training never sees duplicate images
(including, e.g., a source that turns out to mirror another source's rows).

Usage:
    python -m real_data.deduplicate --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \\
        real_data/samples/chanrith_ocr_image_line real_data/samples/darayut_scene_text \\
        real_data/samples/sokheng_synthetic_v1 \\
        --out-dir real_data/samples/dedup
"""
import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from .config import REAL_DATA_ROOT
from .dedup import Record, find_duplicates


def load_full_line_records(root: Path) -> list:
    """Reads root/manifest.tsv and returns Record objects for its
    is_full_line==True rows -- the whole-line images, which are what
    recognizer/ actually trains on (see recognizer/data/manifest.py)."""
    manifest_path = root / "manifest.tsv"
    images_dir = root / "images"
    records = []
    with manifest_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["is_full_line"] != "True":
                continue
            records.append(Record(
                image_path=images_dir / row["filename"],
                text=row["text"],
                source=row["source"],
                line_id=row["line_id"],
            ))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-dirs", nargs="+", required=True,
                         help="One or more real_data/samples/<source> directories to pool and deduplicate.")
    parser.add_argument("--out-dir", type=Path, default=REAL_DATA_ROOT / "samples" / "dedup")
    parser.add_argument("--near-dup-threshold", type=int, default=4,
                         help="Perceptual-hash Hamming distance <= this counts as a near-duplicate. "
                              "0 disables near-duplicate detection (exact-hash dedup only).")
    args = parser.parse_args()

    records = []
    for d in args.real_data_dirs:
        source_records = load_full_line_records(Path(d))
        print(f"{d}: {len(source_records)} whole-line samples")
        records.extend(source_records)

    if not records:
        raise SystemExit("No is_full_line records found under the given --real-data-dirs.")

    print(f"hashing {len(records)} images (exact + near-duplicate detection)...")
    result = find_duplicates(
        list(tqdm(records, desc="records")), near_dup_threshold=args.near_dup_threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["image_path", "text", "source", "line_id"])
        for r in result.kept:
            writer.writerow([str(r.image_path.resolve()), r.text, r.source, r.line_id])

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
    report_path = args.out_dir / "dedup_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\nkept {len(result.kept)}/{len(records)} "
          f"(-{result.exact_duplicate_count} exact, -{result.near_duplicate_count} near-duplicate)")
    print(f"manifest: {manifest_path}")
    print(f"report:   {report_path}")
    if result.duplicate_text_count:
        print(f"note: {result.duplicate_text_count} kept samples share a transcript with another "
              f"kept sample (different image, not removed -- see dedup.py docstring).")


if __name__ == "__main__":
    main()
