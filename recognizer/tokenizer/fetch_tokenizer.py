"""CLI: download the project's shared tokenizer (Panhapich/khmer-sp-8k) --
the SentencePiece model PLUS its khmer_segmentation.py preprocessing module
and the gazetteer/exceptions data files it needs. A bare `khmer_sp.model` is
not enough: encode/decode must go through khmer_segmentation.py's
KhmerTokenizer wrapper (word segmentation + gazetteer/loanword masking) or
tokenization is silently wrong -- see khmer_ocr_tokenizer.py's docstring.

Usage:
    python -m recognizer.tokenizer.fetch_tokenizer --out-dir recognizer/tokenizer/assets
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ..config import TOKENIZER_HUB_ID, TOKENIZER_ASSETS_DIR

REQUIRED_FILES = ("khmer_sp.model", "khmer_segmentation.py", "gazetteer.json")


def fetch_tokenizer(out_dir: Path = TOKENIZER_ASSETS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=TOKENIZER_HUB_ID,
        local_dir=out_dir,
        allow_patterns=["*.model", "*.py", "*.json"],
    )
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Downloaded {TOKENIZER_HUB_ID} into {out_dir}, but these required files "
            f"are missing: {missing}. The upstream repo layout may have changed -- "
            f"inspect {out_dir} and update fetch_tokenizer.py / khmer_ocr_tokenizer.py."
        )
    downloaded = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    print(f"Fetched {TOKENIZER_HUB_ID} -> {out_dir}: {downloaded}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=TOKENIZER_ASSETS_DIR)
    args = parser.parse_args()
    fetch_tokenizer(args.out_dir)


if __name__ == "__main__":
    main()
