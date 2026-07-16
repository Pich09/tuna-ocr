"""Scan-realism augmentations applied to rendered line/page images:
blur, rotation, contrast/brightness jitter, gaussian noise, JPEG artifacts,
and paper-texture-style shading. Works on PIL images; bounding boxes in
pixel space are rotated alongside the image when rotation is applied.
"""
import io
import math
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def random_blur(img: Image.Image, max_radius: float = 1.2) -> Image.Image:
    radius = random.uniform(0, max_radius)
    return img.filter(ImageFilter.GaussianBlur(radius)) if radius > 0.05 else img


def random_brightness_contrast(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.2))
    return img


def gaussian_noise(img: Image.Image, sigma: float = 6.0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def jpeg_artifacts(img: Image.Image, quality_range=(35, 85)) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(*quality_range))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rotate_with_boxes(img: Image.Image, boxes, max_deg: float = 2.0, fill=(255, 255, 255)):
    """boxes: list of (x0, y0, x1, y1) in pixel space. Returns (rotated_img, rotated_boxes)
    where each box is re-derived as the axis-aligned bounding box of the
    rotated corners (safe upper bound, avoids clipped/rotated label boxes).
    """
    angle = random.uniform(-max_deg, max_deg)
    w, h = img.size
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)

    if angle == 0 or not boxes:
        return rotated, boxes

    cx, cy = w / 2, h / 2
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rotate_point(x, y):
        # PIL rotates counter-clockwise for positive angle; invert to map
        # original box corners into the rotated image's coordinate frame.
        dx, dy = x - cx, y - cy
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        return rx + cx, ry + cy

    new_boxes = []
    for x0, y0, x1, y1 in boxes:
        corners = [rotate_point(x, y) for x in (x0, x1) for y in (y0, y1)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        new_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    return rotated, new_boxes


def degrade_image(img: Image.Image, boxes=None, p_rotate: float = 0.3):
    """Applies a randomized subset of degradations. If `boxes` is given
    (pixel-space (x0,y0,x1,y1) list), returns (img, boxes); else returns img.
    """
    if boxes is not None and random.random() < p_rotate:
        img, boxes = rotate_with_boxes(img, boxes)

    img = random_blur(img)
    img = random_brightness_contrast(img)
    if random.random() < 0.5:
        img = gaussian_noise(img)
    if random.random() < 0.4:
        img = jpeg_artifacts(img)

    return (img, boxes) if boxes is not None else img
