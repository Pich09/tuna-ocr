"""CLI: generate a full-page document dataset (book / research paper /
national ID layouts) with YOLO-format detection labels for title/header/
paragraph/line regions, plus per-line transcripts reusable for recognizer
training.

Usage:
    python -m data_gen.generate_pages --out-dir dataset/pages --num-pages 2000
"""
import argparse
import json
import random
from pathlib import Path

import yaml
from tqdm import tqdm

from .config import PageRenderConfig, LAYOUT_CLASSES, LANGUAGES
from .degrade import degrade_image
from .page_layout import TEMPLATES
from .page_renderer import render_page, write_sample
from .text_sampler import TextSampler
from .utils.fonts import FontBank


def write_dataset_yaml(out_dir: Path):
    data = {
        "path": str(out_dir.resolve()),
        "train": "images",
        "val": "images",
        "names": {i: name for i, name in enumerate(LAYOUT_CLASSES)},
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/pages"))
    parser.add_argument("--num-pages", type=int, default=2000)
    parser.add_argument("--primary-lang", default="km", choices=LANGUAGES)
    parser.add_argument("--template", choices=list(TEMPLATES) + [None], default=None,
                         help="Fix a single layout template; default samples randomly per page")
    parser.add_argument("--no-degrade", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    font_bank = FontBank()
    sampler = TextSampler()
    cfg = PageRenderConfig()

    transcripts_path = args.out_dir / "line_transcripts.jsonl"
    with transcripts_path.open("w", encoding="utf-8") as transcripts_f:
        for i in tqdm(range(args.num_pages), desc="rendering pages"):
            sample = render_page(font_bank, sampler, cfg, template=args.template, primary_lang=args.primary_lang)

            if not args.no_degrade:
                boxes = [box for _, box in sample["detections"]]
                img, rotated_boxes = degrade_image(sample["image"], boxes=boxes)
                sample["image"] = img
                sample["detections"] = [
                    (cls, box) for (cls, _), box in zip(sample["detections"], rotated_boxes)
                ]
                # Keep per-line transcript boxes in sync with the (possibly rotated) detections.
                rotated_line_boxes = [box for cls, box in sample["detections"] if cls == "line"]
                for line_record, box in zip(sample["lines"], rotated_line_boxes):
                    line_record["box"] = box

            name = f"page_{i:06d}"
            write_sample(args.out_dir, name, sample, cfg)

            for line in sample["lines"]:
                transcripts_f.write(json.dumps({"page": name, **line}, ensure_ascii=False) + "\n")

    write_dataset_yaml(args.out_dir)
    print(f"Wrote {args.num_pages} pages to {args.out_dir}")


if __name__ == "__main__":
    main()
