"""Batch-processing configuration helpers."""

from copy import deepcopy

import numpy as np


DEFAULT_EXPORT_OPTIONS = {
    "result_images": True,
    "binary_images": False,
    "source_crops": False,
    "summary_excel": True,
    "segment_excel": True,
    "metrics": {
        "area": True,
        "area_ratio": True,
        "length": True,
        "average_width": True,
        "maximum_width": True,
        "junction_count": True,
        "fractal_dimension": True,
        "segment_count": True,
    },
    "segment_fields": {
        "length": True,
        "width": True,
    },
}


def normalize_export_options(options=None):
    merged = deepcopy(DEFAULT_EXPORT_OPTIONS)
    if not options:
        return merged

    for key, value in options.items():
        if key in ("metrics", "segment_fields") and isinstance(value, dict):
            merged[key].update(value)
        elif key in merged:
            merged[key] = bool(value)
    return merged


def normalize_crop_rect(rect, image_shape):
    if rect is None:
        return None
    if len(image_shape) < 2:
        return None

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return None

    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    left = max(0, min(w, left))
    right = max(0, min(w, right))
    top = max(0, min(h, top))
    bottom = max(0, min(h, bottom))

    if right - left < 2 or bottom - top < 2:
        return None
    if left == 0 and top == 0 and right == w and bottom == h:
        return None
    return left, top, right, bottom


def crop_image(image, rect):
    normalized = normalize_crop_rect(rect, image.shape)
    if normalized is None:
        return image.copy()
    left, top, right, bottom = normalized
    return image[top:bottom, left:right].copy()


def normalize_circle_roi(circle, image_shape):
    if circle is None:
        return None
    if len(image_shape) < 2:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return None

    cx, cy, radius = [int(round(v)) for v in circle]
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    max_radius = min(cx, cy, w - 1 - cx, h - 1 - cy)
    radius = max(0, min(int(radius), int(max_radius)))
    if radius < 2:
        return None
    return cx, cy, radius


def create_circle_mask(image_shape, circle):
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    normalized = normalize_circle_roi(circle, (h, w))
    if normalized is None:
        mask[:, :] = 255
        return mask

    cx, cy, radius = normalized
    yy, xx = np.ogrid[:h, :w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    mask[inside] = 255
    return mask


def mask_to_circle_roi(image, circle, outside_value=0):
    mask = create_circle_mask(image.shape, circle)
    if mask is None:
        return image.copy()
    result = np.array(image, copy=True)
    if result.ndim == 2:
        result[mask == 0] = outside_value
    else:
        result[mask == 0] = outside_value
    return result


def build_summary_record(fname, unit, values, options=None):
    options = normalize_export_options(options)
    metrics = options["metrics"]

    record = {"File Name": fname}
    if metrics.get("area"):
        record[f"Area({unit}^2)"] = round(values.get("area", 0), 2)
    if metrics.get("area_ratio"):
        record["Area Ratio (%)"] = round(values.get("area_ratio", 0), 2)
    if metrics.get("length"):
        record[f"Total Length({unit})"] = round(values.get("length", 0), 2)
    if metrics.get("average_width"):
        record[f"Average Width({unit})"] = round(values.get("average_width", 0), 2)
    if metrics.get("maximum_width"):
        record[f"Maximum Width({unit})"] = round(values.get("maximum_width", 0), 2)
    if metrics.get("junction_count"):
        record["Junction Count"] = int(values.get("junction_count", 0))
    if metrics.get("fractal_dimension"):
        record["Fractal Dimension"] = round(values.get("fractal_dimension", 0), 4)
    if metrics.get("segment_count"):
        record["Valid Crack Count"] = int(values.get("segment_count", 0))
    return record


def build_segment_record(fname, segment_id, unit, length, width, options=None):
    options = normalize_export_options(options)
    fields = options["segment_fields"]

    record = {"File Name": fname, "ID": segment_id}
    if fields.get("length"):
        record[f"Length({unit})"] = round(length, 2)
    if fields.get("width"):
        record[f"Width({unit})"] = round(width, 2)
    return record
