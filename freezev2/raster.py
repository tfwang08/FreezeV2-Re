from __future__ import annotations

import numpy as np

from .geometry import project_points
from .onboard import CameraPose


def _sample_rendered_depth(
    depth: np.ndarray,
    raster_xy: np.ndarray,
    sampling: str,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape[:2]
    xy = np.asarray(raster_xy, dtype=np.float64)

    if sampling == "nearest":
        pixels = np.rint(xy).astype(np.int64)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        sampled = np.zeros(len(xy), dtype=np.float64)
        sampled[inside] = depth[pixels[inside, 1], pixels[inside, 0]]
        return sampled, inside & (sampled > 0)

    if sampling not in {"bilinear", "inverse_bilinear"}:
        raise ValueError("sampling must be 'nearest', 'bilinear', or 'inverse_bilinear'")

    inside = (
        (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= w - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= h - 1)
    )
    sampled = np.zeros(len(xy), dtype=np.float64)
    ids = np.flatnonzero(inside)
    if len(ids) == 0:
        return sampled, np.zeros(len(xy), dtype=bool)

    u = xy[ids, 0]
    v = xy[ids, 1]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = u - x0
    wy = v - y0

    values = np.stack(
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
    valid_values = values > 0
    weights = np.where(valid_values, weights, 0.0)
    weight_sum = weights.sum(axis=1)
    has_depth = weight_sum > 0

    local = np.zeros(len(ids), dtype=np.float64)
    if sampling == "bilinear":
        local[has_depth] = (
            (values[has_depth] * weights[has_depth]).sum(axis=1)
            / weight_sum[has_depth]
        )
    else:
        inverse_sum = np.zeros(np.count_nonzero(has_depth), dtype=np.float64)
        valid_values_hd = valid_values[has_depth]
        values_hd = values[has_depth]
        weights_hd = weights[has_depth]
        inverse_terms = np.zeros_like(values_hd)
        inverse_terms[valid_values_hd] = (
            weights_hd[valid_values_hd] / values_hd[valid_values_hd]
        )
        inverse_sum = inverse_terms.sum(axis=1) / weight_sum[has_depth]
        local[has_depth] = 1.0 / inverse_sum

    sampled[ids] = local
    valid = np.zeros(len(xy), dtype=bool)
    valid[ids[has_depth]] = True
    return sampled, valid


def visible_query_image_coordinates(
    query_points: np.ndarray,
    depth: np.ndarray,
    camera: CameraPose,
    depth_tolerance: float,
    sampling: str = "inverse_bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible query ids and continuous rendered-image coordinates.

    ``project_points`` returns OpenGL/window-style image coordinates for the BOP
    VisPy renderer. Fragment centers are at half-integer window coordinates,
    while NumPy depth-array centers are indexed by integers. Therefore rendered
    depth is sampled at ``uv - 0.5``. The returned coordinates remain the
    original continuous ``uv`` because FoundPose-style ``grid_sample`` with
    ``align_corners=False`` expects pixel centers at ``j + 0.5``.
    """
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")

    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")

    uv, z = project_points(query_points, camera.K, camera.R, camera.t)
    finite = np.isfinite(uv).all(axis=1) & (z > 0)
    ids = np.flatnonzero(finite)
    if len(ids) == 0:
        return ids, np.empty((0, 2), dtype=np.float64)

    raster_xy = uv[ids] - 0.5
    rendered_z, has_depth = _sample_rendered_depth(
        depth,
        raster_xy,
        sampling=sampling,
    )
    visible = has_depth & (np.abs(rendered_z - z[ids]) <= depth_tolerance)
    visible_ids = ids[visible]
    return visible_ids, uv[visible_ids].copy()
