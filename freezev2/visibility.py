from __future__ import annotations

from typing import Sequence

import numpy as np

from .onboard import Template
from .raster import visible_query_image_coordinates


def query_visibility_counts(
    query_points: np.ndarray,
    templates: Sequence[Template],
    depth_tolerance: float,
    feature_image_hws: Sequence[tuple[int, int]] | None = None,
    sampling: str = "inverse_bilinear",
) -> np.ndarray:
    """Count in how many rendered views each query point is visible.

    Rendered-depth lookup follows the BOP/VisPy fragment-center convention:
    projected window coordinates correspond to half-integer array centers, so
    depth sampling is performed at ``uv - 0.5`` by
    :func:`visible_query_image_coordinates`.

    ``feature_image_hws`` optionally restricts visibility to the image region
    consumed by the visual backbone. For ViT-g/14 on the paper's 480x480
    templates this is 476x476.
    """
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")
    if feature_image_hws is not None and len(feature_image_hws) != len(templates):
        raise ValueError("feature_image_hws must contain one size per template")

    counts = np.zeros(len(query_points), dtype=np.int32)
    for view_id, template in enumerate(templates):
        ids, image_xy = visible_query_image_coordinates(
            query_points,
            template.depth,
            template.camera,
            depth_tolerance=depth_tolerance,
            sampling=sampling,
        )
        if len(ids) == 0:
            continue

        if feature_image_hws is not None:
            image_h, image_w = map(int, feature_image_hws[view_id])
            inside = (
                (image_xy[:, 0] >= 0.0)
                & (image_xy[:, 0] < image_w)
                & (image_xy[:, 1] >= 0.0)
                & (image_xy[:, 1] < image_h)
            )
            ids = ids[inside]

        counts[ids] += 1

    return counts


def select_visible_query_points(
    raw_query_points: np.ndarray,
    templates: Sequence[Template],
    depth_tolerance: float,
    final_count: int = 5000,
    min_views: int = 18,
    feature_image_hws: Sequence[tuple[int, int]] | None = None,
    sampling: str = "inverse_bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a raw Poisson cloud by visibility and keep exactly ``final_count``.

    FreeZeV2 specifies a raw query cloud, a visibility threshold of 18 views,
    and a final 5k-point query cloud, but does not publish ``N_Q^raw``. The raw
    count therefore stays an explicit caller-side reproduction parameter.

    Selection preserves the original raw Poisson order rather than ranking by
    visibility, which would bias the final cloud toward easy-to-see surfaces.
    If too few points survive, the caller must increase the explicit raw count;
    this function never relaxes ``min_views`` or ``depth_tolerance``.
    """
    raw_query_points = np.asarray(raw_query_points, dtype=np.float64)
    if raw_query_points.ndim != 2 or raw_query_points.shape[1] != 3:
        raise ValueError("raw_query_points must have shape Nx3")
    if final_count <= 0:
        raise ValueError("final_count must be positive")
    if min_views <= 0:
        raise ValueError("min_views must be positive")

    counts = query_visibility_counts(
        raw_query_points,
        templates,
        depth_tolerance=depth_tolerance,
        feature_image_hws=feature_image_hws,
        sampling=sampling,
    )
    visible_ids = np.flatnonzero(counts >= int(min_views))
    if len(visible_ids) < int(final_count):
        raise ValueError(
            f"only {len(visible_ids)} raw query points are visible in at least "
            f"{min_views} views; need {final_count}"
        )

    selected_ids = visible_ids[: int(final_count)]
    return raw_query_points[selected_ids].copy(), counts[selected_ids].copy()
