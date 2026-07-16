"""
plot_layout_tags.py
--------------------
Draws YOLO-format layout-detection labels back onto their image, color-coded
by class with a legend, so you can visually sanity-check a generated sample
without loading it into a training framework. Works on the output of any
generator that follows this repo's convention (one class id per line in
LAYOUT_CLASSES order, `<cls> xc yc w h` normalized to [0, 1]) --
generate_pages.py and generate_letter_v2.py both qualify.

Run:
    python -m data_gen.plot_layout_tags --dataset-dir data_gen/samples/letterhead_dataset --name letter_00000
    python -m data_gen.plot_layout_tags --image path/to/img.jpg --label path/to/img.txt --out preview.png
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import LAYOUT_CLASSES

COLORS = {
    "title": (220, 20, 20),
    "header": (20, 110, 220),
    "paragraph": (20, 160, 20),
    "line": (235, 140, 0),
    "field": (160, 0, 160),
    "signature": (30, 30, 30),
    "photo": (0, 180, 180),
    "stamp": (180, 0, 0),
}


def plot_tags(image_path: Path, label_path: Path, out_path: Path, classes=LAYOUT_CLASSES):
    """Draws every box in `label_path` onto a copy of `image_path`, with a
    class-color legend in the top-left corner, and saves it to `out_path`."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    rows = [
        line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for cls_id, xc, yc, bw, bh in rows:
        name = classes[int(cls_id)]
        color = COLORS.get(name, (255, 0, 255))
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x0, x1 = (xc - bw / 2) * w, (xc + bw / 2) * w
        y0, y1 = (yc - bh / 2) * h, (yc + bh / 2) * h
        width = 3 if name in ("title", "header", "stamp") else 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    used = sorted({classes[int(r[0])] for r in rows}, key=classes.index) or classes
    lx, ly = 20, 20
    draw.rectangle([lx - 8, ly - 8, lx + 180, ly + 8 + 18 * len(used)],
                   fill=(255, 255, 255), outline=(0, 0, 0))
    for i, name in enumerate(used):
        yy = ly + i * 18
        draw.rectangle([lx, yy, lx + 14, yy + 10], fill=COLORS.get(name, (255, 0, 255)))
        draw.text((lx + 20, yy - 2), name, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                         help="dataset root with images/<name>.jpg + labels/<name>.txt")
    parser.add_argument("--name", help="sample name (without extension); default: first found")
    parser.add_argument("--image", type=Path, help="path to a single image (alternative to --dataset-dir)")
    parser.add_argument("--label", type=Path, help="path to that image's YOLO .txt label")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    args = parser.parse_args()

    if args.dataset_dir:
        images_dir = args.dataset_dir / "images"
        labels_dir = args.dataset_dir / "labels"
        if args.name:
            name = args.name
        else:
            name = sorted(p.stem for p in images_dir.glob("*.jpg"))[0]
        image_path = images_dir / f"{name}.jpg"
        label_path = labels_dir / f"{name}.txt"
        out_path = args.out or (args.dataset_dir / f"{name}_tags_preview.png")
    elif args.image and args.label:
        image_path, label_path = args.image, args.label
        out_path = args.out or image_path.with_name(f"{image_path.stem}_tags_preview.png")
    else:
        parser.error("pass either --dataset-dir [--name ...] or both --image and --label")
        return

    out = plot_tags(image_path, label_path, out_path)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
