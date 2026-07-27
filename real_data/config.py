"""Central config for the real (non-synthetic) external data pipeline."""
from dataclasses import dataclass
from pathlib import Path

REAL_DATA_ROOT = Path(__file__).resolve().parent

# Holds the packed+deduplicated dataset (see pack_arrow.py, deduplicate.py,
# hf_push.py) -- lets the pull->pack->dedup pipeline run once and be reused
# across sessions/notebooks instead of repeating it every time.
HF_DATA_REPO_ID = "Panhapich/tuna-ocr-data"


@dataclass
class ExternalChunkConfig:
    """Fixed-width overlapping-chunk windowing for external-source line
    images before they reach a Conformer-style encoder (see
    real_data/chunking.py). These are placeholders until the actual
    encoder's block size / subsampling factor is chosen -- not tuned
    values.
    """
    chunk_width: int = 256
    overlap: int = 32


# External Hugging Face OCR datasets to sample from (see
# real_data/hf_datasets.py). `image_col`/`text_col` of None means "run
# column auto-detection" -- all 4 of these have undocumented/unconfirmed
# schemas, so all rely on auto-detection rather than a trusted column name.
#
# SoyVitou/khmer-handwritten-dataset-4.2k was removed on purpose: it was the
# only handwritten source in the mix, and training is focused on printed
# text first -- add a dedicated handwritten-data pass back in later rather
# than blending a single handwritten source in with printed ones now.
EXTERNAL_DATASETS = {
    "deepcopy_khmer_text_recognition": {
        "hub_id": "deepcopy/khmer-text-recognition",
        "split": "train",
        "image_col": None,
        "text_col": None,
    },
    "chanrith_ocr_image_line": {
        "hub_id": "Chanrith123333/khmer_english_ocr_image_line",
        "split": "train",
        "image_col": None,
        "text_col": None,
    },
    "darayut_scene_text": {
        "hub_id": "Darayut/khmer-scene-text-synthetic-contrast",
        "split": "train",
        "image_col": None,
        "text_col": None,
    },
    "sokheng_synthetic_v1": {
        "hub_id": "Sokheng/khmer-synthetic-ocr-v1-100k",
        "split": "train",
        "image_col": None,
        "text_col": None,
    },
}
