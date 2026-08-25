from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import project_points
from .onboard import Template, map_visible_pixels_to_query_points


def _subpixel_visible_ids(
    query_points: np.ndarray,
    template: Template,
    depth_tolerance: float,
    image_hw: tuple[int, int],
    inverse_depth: bool,
) -> np.ndarray:
    """Return visible point ids using sub-pixel rendered-depth interpolation."""
    depth = np.asarray(template.depth, dtype=np.float64)
    image_h = min(int(image_hw[0]), depth.shape[0])
    image_w = min(int(image_hw[1]), depth.shape[1])

    uv, z = project_points(
        query_points,
        template.camera.K,
        template.camera.R,
        template.camera.t,
    )
    finite = np.isfinite(uv).all(axis=1)
    inside = (
        finite
        & (z > 0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= image_w - 1)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= image_h - 1)
    )
    ids = np.flatnonzero(inside)
    if len(ids) == 0:
        return ids

    u = uv[ids, 0]
    v = uv[ids, 1]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, image_w - 1)
    y1 = np.minimum(y0 + 1, image_h - 1)
    wx = u - x0
    wy = v - y0

    samples = np.stack(
        [
            depth[y0, x0],
            depth[y0, x1],
            depth[y1, x0],
            depth[y1, x1],
        ],
        axis=1,
    )
    weights = np.stack(
        [
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ],
        axis=1,
    )

    # Background depth is zero. Ignore such neighbours and renormalize the
    # remaining interpolation weights so silhouettes do not mix with zero.
    valid_depth = samples > 0
    weights = np.where(valid_depth, weights, 0.0)
    weight_sum = weights.sum(axis=1)
    has_depth = weight_sum > 0
    rendered_z = np.zeros(len(ids), dtype=np.float64)

    if inverse_depth:
        inverse_samples = np.zeros_like(samples)
        inverse_samples[valid_depth] = 1.0 / samples[valid_depth]
        rendered_inv_z = np.zeros(len(ids), dtype=np.float64)
        rendered_inv_z[has_depth] = (
            (inverse_samples[has_depth] * weights[has_depth]).sum(axis=1)
            / weight_sum[has_depth]
        )
        positive = has_depth & (rendered_inv_z > 0)
        rendered_z[positive] = 1.0 / rendered_inv_z[positive]
        has_depth = positive
    else:
        rendered_z[has_depth] = (
            (samples[has_depth] * weights[has_depth]).sum(axis=1)
            / weight_sum[has_depth]
        )

    visible = has_depth & (np.abs(rendered_z - z[ids]) <= depth_tolerance)
    return ids[visible]


def query_visibility_counts(
    query_points: np.ndarray,
    templates: Sequence[Template],
    depth_tolerance: float,
    feature_image_hws: Sequence[tuple[int, int]] | None = None,
    sampling: str = "nearest",
) -> np.ndarray:
    """Count in how many rendered views each query point is visible.

    ``sampling='nearest'`` preserves the original diagnostic based on the
    rounded projected pixel. ``sampling='bilinear'`` interpolates metric depth
    at the continuous projected coordinate. ``sampling='inverse_bilinear'``
    interpolates ``1 / depth`` before inverting back to metric z, matching the
    perspective-correct depth behavior of a planar triangle more closely.

    Sub-pixel modes ignore zero-depth neighbours and renormalize the remaining
    weights. These modes are diagnostic only; the production feature aggregation
    path is unchanged until the geometry is validated on real templates.

    ``feature_image_hws`` optionally restricts visibility to the image region
    actually consumed by the visual backbone. For ViT-g/14 on the paper's
    480x480 templates this is 476x476, so the dropped right/bottom border is not
    counted as a DINO observation.
    """
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")
    if feature_image_hws is not None and len(feature_image_hws) != len(templates):
        raise ValueError("feature_image_hws must contain one size per template")
    if sampling not in {"nearest", "bilinear", "inverse_bilinear"}:
        raise ValueError(
            "sampling must be 'nearest', 'bilinear', or 'inverse_bilinear'"
        )

    counts = np.zeros(len(query_points), dtype=np.int32)
    for view_id, template in enumerate(templates):
        image_hw = (
            tuple(map(int, template.depth.shape[:2]))
            if feature_image_hws is None
            else tuple(map(int, feature_image_hws[view_id]))
        )

        if sampling in {"bilinear", "inverse_bilinear"}:
            ids = _subpixel_visible_ids(
                query_points,
                template,
                depth_tolerance=depth_tolerance,
                image_hw=image_hw,
                inverse_depth=sampling == "inverse_bilinear",
            )
        else:
            ids, pixels = map_visible_pixels_to_query_points(
                query_points,
                template.depth,
                template.camera,
                depth_tolerance=depth_tolerance,
            )
            if len(ids) and feature_image_hws is not None:
                image_h, image_w = image_hw
                inside = (
                    (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < image_w)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < image_h)
                )
                ids = ids[inside]

        counts[ids] += 1

    return counts
