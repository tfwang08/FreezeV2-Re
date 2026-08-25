from __future__ import annotations

from typing import Sequence

import numpy as np

from .features import sample_feature_map
from .onboard import Template
from .raster import visible_query_image_coordinates


def _view_weight_matrix(view_weights, num_views: int, num_points: int) -> np.ndarray:
    if view_weights is None:
        return np.ones((num_views, num_points), dtype=np.float64)

    weights = np.asarray(view_weights, dtype=np.float64)
    if weights.ndim == 1:
        if weights.shape != (num_views,):
            raise ValueError("1D view_weights must contain one value per view")
        weights = np.broadcast_to(weights[:, None], (num_views, num_points))
    elif weights.shape != (num_views, num_points):
        raise ValueError("view_weights must have shape [views] or [views, points]")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("view_weights must be finite and non-negative")
    return weights


def aggregate_query_visual_features_streaming(
    query_points: np.ndarray,
    templates: Sequence[Template],
    extractor,
    depth_tolerance: float,
    min_views: int = 18,
    view_weights=None,
    depth_sampling: str = "inverse_bilinear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode rendered views one at a time and aggregate query visual features.

    Visibility uses the BOP/VisPy pixel-center convention implemented in
    :func:`visible_query_image_coordinates`: rendered depth is sampled at
    ``projected_uv - 0.5``. DINO features are sampled at the original continuous
    projected coordinates, matching FoundPose-style ``grid_sample`` with
    ``align_corners=False``.

    ``depth_sampling`` is an explicit reproduction choice because FreeZeV2 does
    not publish its exact rendered-depth visibility implementation.
    """
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if min_views <= 0:
        raise ValueError("min_views must be positive")
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")
    if len(templates) == 0:
        return (
            query_points[:0],
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    weights = _view_weight_matrix(view_weights, len(templates), len(query_points))
    sums = None
    weight_sums = np.zeros(len(query_points), dtype=np.float64)
    counts = np.zeros(len(query_points), dtype=np.int32)

    for view_id, template in enumerate(templates):
        feature_map = extractor.encode(template.rgb)
        if getattr(feature_map, "ndim", None) != 3:
            raise ValueError("DINO extractor must return a CxHxW feature map")
        if sums is None:
            sums = np.zeros(
                (len(query_points), int(feature_map.shape[0])), dtype=np.float64
            )
        elif int(feature_map.shape[0]) != sums.shape[1]:
            raise ValueError("all DINO views must have the same channel count")

        image_hw = getattr(extractor, "last_image_hw", None)
        if image_hw is None:
            image_hw = tuple(map(int, template.depth.shape[:2]))
        image_h, image_w = map(int, image_hw)

        ids, image_xy = visible_query_image_coordinates(
            query_points,
            template.depth,
            template.camera,
            depth_tolerance=depth_tolerance,
            sampling=depth_sampling,
        )
        if len(ids) == 0:
            del feature_map
            continue

        inside = (
            (image_xy[:, 0] >= 0.0)
            & (image_xy[:, 0] < image_w)
            & (image_xy[:, 1] >= 0.0)
            & (image_xy[:, 1] < image_h)
        )
        ids = ids[inside]
        image_xy = image_xy[inside]
        if len(ids) == 0:
            del feature_map
            continue

        sampled = sample_feature_map(
            feature_map,
            image_xy,
            image_hw=(image_h, image_w),
        )
        sampled_np = sampled.detach().to("cpu").numpy().astype(np.float64, copy=False)
        point_weights = weights[view_id, ids]
        sums[ids] += sampled_np * point_weights[:, None]
        weight_sums[ids] += point_weights
        counts[ids] += 1
        del feature_map

    assert sums is not None
    keep = (counts >= int(min_views)) & (weight_sums > 0)
    descriptors = sums[keep] / weight_sums[keep, None]
    return (
        query_points[keep].copy(),
        descriptors.astype(np.float32),
        counts[keep].copy(),
    )
