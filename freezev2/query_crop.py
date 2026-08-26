from __future__ import annotations

import numpy as np


QUERY_DINO_INPUT_SIZE = 224
QUERY_DINO_PATCH_GRID = 16
QUERY_DINO_MODE = "tight_mask_crop_224_pixel_lift"


def query_object_crop_bbox(mask: np.ndarray) -> np.ndarray:
    """Return the tight axis-aligned extent occupied by a rendered object mask.

    FreeZe crops each rendered query image to the portion occupied by the object
    before feeding the RGB crop to DINOv2. Bounds are integer pixel-cell extents
    in ``[x0, y0, x1, y1]`` form with an exclusive upper edge.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape HxW")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("query object mask is empty")
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.int64,
    )


def resize_query_object_crop(
    rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_size: int = QUERY_DINO_INPUT_SIZE,
) -> np.ndarray:
    """Crop the rendered object extent and resize it to the DINO input canvas."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to resize query DINO crops") from exc

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    bbox = np.asarray(bbox_xyxy, dtype=np.int64).reshape(4)
    x0, y0, x1, y1 = map(int, bbox)
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("bbox_xyxy must be a positive extent inside the image")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    image = Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB")
    crop = image.crop((x0, y0, x1, y1)).resize(
        (output_size, output_size),
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(crop, dtype=np.uint8).copy()


def renderer_pixels_to_query_crop_xy(
    pixels_xy: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_size: int = QUERY_DINO_INPUT_SIZE,
) -> np.ndarray:
    """Map renderer pixel centers to continuous coordinates in the resized crop.

    Renderer pixels are represented by integer array indices, while the BOP/VisPy
    image-space center of pixel ``(u, v)`` is ``(u + 0.5, v + 0.5)``. After a
    tight crop and resize, these continuous centers are scaled into the 224x224
    DINO coordinate system used by ``sample_feature_map``.
    """
    pixels = np.asarray(pixels_xy)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_xy must have shape Nx2")
    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if not np.isfinite(bbox).all() or width <= 0 or height <= 0:
        raise ValueError("bbox_xyxy must have positive finite extent")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    centers = pixels.astype(np.float64) + 0.5
    xy = np.empty_like(centers, dtype=np.float64)
    xy[:, 0] = (centers[:, 0] - x0) * output_size / width
    xy[:, 1] = (centers[:, 1] - y0) * output_size / height
    return xy
