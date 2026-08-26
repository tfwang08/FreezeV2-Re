from __future__ import annotations

import numpy as np


TARGET_DINO_INPUT_SIZE = 224
TARGET_DINO_PATCH_GRID = 16
TARGET_DINO_MODE = "square_crop_224_direct_tokens"


def target_patch_grid(
    mask: np.ndarray,
    grid_size: int = TARGET_DINO_PATCH_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return retained square-bbox patch centers and their direct token ids.

    The candidate mask is enclosed by the smallest axis-aligned square. A
    ``grid_size x grid_size`` grid is laid over that square and only patch
    centers whose containing image cell is inside the mask are retained.

    ``token_indices`` are row-major ids into a DINO feature map with the same
    grid shape. This keeps the 2D patch-center geometry and the visual token
    assignment explicit and avoids any post-DINO per-pixel interpolation.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape HxW")
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.full((4,), np.nan, dtype=np.float32),
        )

    x0 = float(xs.min())
    x1 = float(xs.max() + 1)
    y0 = float(ys.min())
    y1 = float(ys.max() + 1)
    side = max(x1 - x0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    square_x0 = cx - 0.5 * side
    square_y0 = cy - 0.5 * side
    square_x1 = square_x0 + side
    square_y1 = square_y0 + side
    bbox = np.asarray(
        [square_x0, square_y0, square_x1, square_y1],
        dtype=np.float32,
    )

    step = side / grid_size
    axis = np.arange(grid_size, dtype=np.float64) + 0.5
    x_centers = square_x0 + axis * step
    y_centers = square_y0 + axis * step
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    centers = np.stack((xx.ravel(), yy.ravel()), axis=1)
    pixels = np.floor(centers).astype(np.int64)
    token_indices = np.arange(grid_size * grid_size, dtype=np.int64)

    h, w = mask.shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    keep = np.zeros(len(centers), dtype=bool)
    ids = np.flatnonzero(inside)
    if len(ids):
        keep[ids] = mask[pixels[ids, 1], pixels[ids, 0]]

    return (
        centers[keep].astype(np.float32),
        pixels[keep],
        token_indices[keep],
        bbox,
    )


def resize_square_bbox_crop(
    rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_size: int = TARGET_DINO_INPUT_SIZE,
) -> np.ndarray:
    """Resample a continuous square image extent onto a fixed DINO canvas."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to resize target DINO crops") from exc

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    if not np.isfinite(bbox).all():
        raise ValueError("bbox_xyxy must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("bbox_xyxy must have positive extent")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    image = Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB")
    # EXTENT supports floating-point source bounds and pads samples outside the
    # image canvas. The same continuous bounds define the patch centers used for
    # depth backprojection, so direct feature-token ids stay geometrically tied
    # to the original image square.
    crop = image.transform(
        (output_size, output_size),
        Image.Transform.EXTENT,
        data=tuple(float(v) for v in bbox),
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(crop, dtype=np.uint8).copy()


def direct_feature_map_tokens(
    feature_map,
    grid_size: int = TARGET_DINO_PATCH_GRID,
) -> np.ndarray:
    """Flatten a CxGxG DINO feature map into row-major direct patch tokens."""
    if hasattr(feature_map, "detach"):
        array = feature_map.detach().to("cpu").numpy()
    else:
        array = np.asarray(feature_map)
    array = np.asarray(array, dtype=np.float32)
    grid_size = int(grid_size)
    expected = (grid_size, grid_size)
    if array.ndim != 3 or tuple(array.shape[1:]) != expected:
        raise ValueError(
            "target DINO feature map must have shape CxGxG; "
            f"got {array.shape}, expected (*, {grid_size}, {grid_size})"
        )
    if not np.isfinite(array).all():
        raise ValueError("target DINO feature map contains non-finite values")
    return np.transpose(array, (1, 2, 0)).reshape(grid_size * grid_size, array.shape[0])
