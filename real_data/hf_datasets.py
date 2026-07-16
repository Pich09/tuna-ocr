"""Streams a small sample of line-image OCR examples from external Hugging
Face datasets (see config.EXTERNAL_DATASETS). Network access + the
`datasets` library are required; each dataset entry point fails with a
clear, isolated error rather than crashing a multi-dataset run.

None of these datasets provide per-character/word boxes -- only one
whole-line transcript per image -- so this module has nothing to do with
re-deriving per-chunk labels. It just yields (image, text, source) triples;
chunking (real_data.chunking) is a separate, later step.
"""
import itertools
from typing import Iterator

from PIL import Image

from .config import EXTERNAL_DATASETS


class ExternalDatasetError(Exception):
    def __init__(self, name: str, hub_id: str, reason: str):
        self.name = name
        self.hub_id = hub_id
        self.reason = reason
        super().__init__(
            f"Failed to load '{name}' ({hub_id}): {reason}. Skipping this source. "
            f"If this is a gated dataset, run `huggingface-cli login` first."
        )


def _detect_columns(name: str, first_example: dict):
    image_col = None
    text_col = None
    for key, value in first_example.items():
        if isinstance(value, Image.Image) and image_col is None:
            image_col = key
    for key, value in first_example.items():
        if key == image_col:
            continue
        if isinstance(value, str) and text_col is None:
            text_col = key

    if image_col is None or text_col is None:
        raise ExternalDatasetError(
            name, EXTERNAL_DATASETS[name]["hub_id"],
            f"could not auto-detect image/text columns from {list(first_example.keys())} "
            f"(image_col={image_col}, text_col={text_col}); add an explicit override in "
            f"config.EXTERNAL_DATASETS",
        )
    return image_col, text_col


def load_external_samples(name: str, max_samples: int = 150, seed: int = 0, shuffle_buffer: int = 0) -> Iterator[dict]:
    """Yields {"image": PIL.Image, "text": str, "source": name} dicts.

    Always streams (never full-materializes) -- mrrtmob alone is ~12.1M
    rows. `shuffle_buffer` defaults to 0 (no shuffling, samples come out in
    the dataset's natural streamed order): a streaming shuffle buffer must
    be fully filled, one network-fetched row at a time, before anything is
    yielded at all, so its size directly multiplies startup latency rather
    than just adding a fixed cost -- these datasets embed images directly
    in their parquet rows, and a single row can take on the order of tens
    of seconds to fetch over a slow connection, so even a small buffer
    (e.g. 16) can turn into minutes of wait before the first sample. Pass
    `shuffle_buffer > 0` explicitly if you have a fast connection and want
    less order-correlated sampling.
    """
    if name not in EXTERNAL_DATASETS:
        raise ValueError(f"Unknown external dataset '{name}', choices: {list(EXTERNAL_DATASETS)}")
    entry = EXTERNAL_DATASETS[name]
    hub_id, split = entry["hub_id"], entry["split"]

    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ExternalDatasetError(name, hub_id, f"`datasets` package not installed ({e})")

    try:
        ds = hf_datasets.load_dataset(hub_id, split=split, streaming=True)
        if shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        it = iter(ds)
    except Exception as e:
        raise ExternalDatasetError(name, hub_id, str(e))

    try:
        first = next(it)
    except StopIteration:
        raise ExternalDatasetError(name, hub_id, "dataset stream produced no examples")
    except Exception as e:
        raise ExternalDatasetError(name, hub_id, str(e))

    image_col, text_col = entry["image_col"], entry["text_col"]
    if image_col is None or text_col is None:
        image_col, text_col = _detect_columns(name, first)

    def to_sample(example):
        return {"image": example[image_col], "text": example[text_col], "source": name}

    try:
        yield to_sample(first)
        for example in itertools.islice(it, max_samples - 1):
            yield to_sample(example)
    except Exception as e:
        raise ExternalDatasetError(name, hub_id, str(e))
