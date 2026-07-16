"""CLI: generate a text-line recognition dataset (image + transcript pairs).

Usage:
    python -m data_gen.generate_lines --out-dir dataset/lines --num-samples 5000
"""
import argparse
import csv
import random
from pathlib import Path

from tqdm import tqdm

from .config import LineRenderConfig, LANGUAGES
from .degrade import degrade_image
from .line_renderer import render_line
from .text_sampler import TextSampler
from .utils.fonts import FontBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/lines"))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--primary-lang", default="km", choices=LANGUAGES)
    parser.add_argument("--no-degrade", action="store_true", help="Skip blur/noise/jpeg augmentation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    font_bank = FontBank()
    sampler = TextSampler()
    cfg = LineRenderConfig()

    manifest_path = args.out_dir / "manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["filename", "text", "lang_tags", "primary_lang"])

        for i in tqdm(range(args.num_samples), desc="rendering lines"):
            text, spans = sampler.sample_line(args.primary_lang)
            img, meta = render_line(text, args.primary_lang, font_bank, cfg)
            if not args.no_degrade:
                img = degrade_image(img)

            fname = f"line_{i:07d}.jpg"
            img.save(images_dir / fname, quality=92)

            lang_tags = " ".join(f"<{lang}>{txt}" for lang, txt in spans)
            writer.writerow([fname, text, lang_tags, args.primary_lang])

    print(f"Wrote {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
