"""Bounding-box helpers shared by line/page renderers."""


def xyxy_to_yolo(box, img_w, img_h):
    """(x0, y0, x1, y1) pixel box -> (xc, yc, w, h) normalized to [0, 1]."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    xc = x0 + w / 2
    yc = y0 + h / 2
    return xc / img_w, yc / img_h, w / img_w, h / img_h


def union(boxes):
    """Smallest (x0, y0, x1, y1) box enclosing a list of (x0, y0, x1, y1) boxes."""
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return min(xs0), min(ys0), max(xs1), max(ys1)
