from __future__ import annotations

from typing import Sequence

import numpy as np

from .features import sample_feature_map
from .onboard import Template
from .query_crop import (
    QUERY_DINO_INPUT_SIZE,
    QUERY_DINO_MODE,
    QUERY_DINO_PATCH_GRID,
    query_object_crop_bbox,
    renderer_pixels_to_query_crop_xy,
    resize_query_object_crop,
)
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


def rendered_pixels_to_model_points(
    pixels_xy: np.ndarray,
    depths: np.ndarray,
    camera,
) -> np.ndarray:
    """Backproject rendered foreground pixels and transform them to model space.

    BOP/VisPy depth-array indices refer to raster cells whose image-space centers
    are at ``(u + 0.5, v + 0.5)``. FreeZe first turns each rendered depth image
    into a viewpoint-dependent point cloud and then establishes 3D nearest-
    neighbour correspondences with the query surface. This helper implements
    exactly that geometric lifting step.
    """
    pixels = np.asarray(pixels_xy)
    depths = np.asarray(depths, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_xy must have shape Nx2")
    if depths.shape != (len(pixels),):
        raise ValueError("depths must contain one value per pixel")
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        raise ValueError("depths must be positive and finite")

    K = np.asarray(camera.K, dtype=np.float64)
    R = np.asarray(camera.R, dtype=np.float64)
    t = np.asarray(camera.t, dtype=np.float64).reshape(3)
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError("camera K and R must be 3x3")
    if K[0, 0] == 0 or K[1, 1] == 0:
        raise ValueError("camera focal lengths must be non-zero")

    image_xy = pixels.astype(np.float64) + 0.5
    z = depths
    x = (image_xy[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (image_xy[:, 1] - K[1, 2]) * z / K[1, 1]
    camera_points = np.stack((x, y, z), axis=1)

    # Row-vector inverse of p_cam = p_model @ R.T + t.
    model_points = (camera_points - t[None, :]) @ R
    return model_points


def finalize_pixel_lifted_visual_aggregation(
    query_points: np.ndarray,
    view_feature_sum: np.ndarray,
    pixel_feature_sum: np.ndarray,
    view_counts: np.ndarray,
    pixel_counts: np.ndarray,
    min_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finalize paper-style view means and a pixel-support weighted candidate."""
    query_points = np.asarray(query_points, dtype=np.float64)
    view_feature_sum = np.asarray(view_feature_sum, dtype=np.float64)
    pixel_feature_sum = np.asarray(pixel_feature_sum, dtype=np.float64)
    view_counts = np.asarray(view_counts)
    pixel_counts = np.asarray(pixel_counts)
    min_views = int(min_views)

    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if view_feature_sum.ndim != 2 or pixel_feature_sum.shape != view_feature_sum.shape:
        raise ValueError("feature sums must be matching NxD arrays")
    if len(view_feature_sum) != len(query_points):
        raise ValueError("feature sums must match query_points")
    if view_counts.shape != (len(query_points),):
        raise ValueError("view_counts must have shape N")
    if pixel_counts.shape != (len(query_points),):
        raise ValueError("pixel_counts must have shape N")
    if min_views <= 0:
        raise ValueError("min_views must be positive")

    keep = (
        (view_counts >= min_views)
        & (view_counts > 0)
        & (pixel_counts > 0)
    )
    view_uniform = (
        view_feature_sum[keep] / view_counts[keep, None]
    ).astype(np.float32)
    pixel_support = (
        pixel_feature_sum[keep] / pixel_counts[keep, None]
    ).astype(np.float32)
    return (
        query_points[keep].copy(),
        view_uniform,
        pixel_support,
        view_counts[keep].astype(np.int32, copy=True),
        pixel_counts[keep].astype(np.int64, copy=True),
    )


def _nearest_query_indices_torch(model_points: np.ndarray, query_points_t):
    import torch

    points_t = torch.as_tensor(
        model_points,
        dtype=query_points_t.dtype,
        device=query_points_t.device,
    )
    if len(points_t) == 0:
        return torch.empty((0,), dtype=torch.long, device=query_points_t.device)

    point_sq = (points_t * points_t).sum(dim=1, keepdim=True)
    query_sq = (query_points_t * query_points_t).sum(dim=1).unsqueeze(0)
    distances_sq = point_sq + query_sq - 2.0 * (points_t @ query_points_t.T)
    return torch.argmin(distances_sq, dim=1)


def aggregate_query_visual_features_pixel_lifting_streaming(
    query_points: np.ndarray,
    templates: Sequence[Template],
    extractor,
    min_views: int = 18,
    pixel_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate query visual descriptors through rendered pixel lifting.

    This follows the published FreeZe query-processing direction: DINO patch
    features are interpreted at pixel resolution, foreground rendered pixels are
    backprojected with depth to a viewpoint-dependent 3D point cloud, and each
    lifted pixel is associated to the nearest point of ``P_Q`` in model space.

    Two aggregations are returned from the same expensive DINO pass:

    * ``view_uniform``: each view contributes the mean feature of its pixels
      assigned to a query point, and visible views are averaged uniformly. This
      is the closest explicit baseline to the arithmetic multi-view aggregation
      published in FreeZe 2024.
    * ``pixel_support``: all assigned pixels are averaged directly, equivalently
      weighting each view by its number of supporting pixels. FreeZeV2 states
      that a weighted average is used but does not publish the weighting rule,
      so this branch is intentionally retained only as a reverse-engineering
      candidate.

    Following FreeZe, each rendered object mask first defines a tight RGB crop;
    that crop is resized to 224x224 and encoded into the published 16x16 DINO
    patch grid. Foreground pixels remain in the original renderer frame for depth
    lifting, while their centers are mapped into the resized crop to sample the
    bilinearly upsampled visual features. The dense pixel feature tensor is not
    materialized: chunked ``grid_sample`` is mathematically equivalent to the
    published bilinear patch-to-pixel interpolation while keeping memory bounded.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install feature dependencies with: pip install -e '.[features]'"
        ) from exc

    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if len(query_points) == 0:
        raise ValueError("query_points must be non-empty")
    if min_views <= 0:
        raise ValueError("min_views must be positive")
    pixel_chunk_size = int(pixel_chunk_size)
    if pixel_chunk_size <= 0:
        raise ValueError("pixel_chunk_size must be positive")
    if len(templates) == 0:
        return (
            query_points[:0],
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
        )

    query_points_t = None
    view_feature_sum = None
    pixel_feature_sum = None
    view_counts_t = None
    pixel_counts_t = None
    feature_dim = None

    for template in templates:
        crop_bbox = query_object_crop_bbox(template.mask)
        dino_rgb = resize_query_object_crop(
            template.rgb,
            crop_bbox,
            output_size=QUERY_DINO_INPUT_SIZE,
        )
        feature_map = extractor.encode(dino_rgb)
        if not torch.is_tensor(feature_map):
            feature_map = torch.as_tensor(feature_map)
        if feature_map.ndim != 3:
            raise ValueError("DINO extractor must return a CxHxW feature map")
        if not feature_map.is_floating_point():
            feature_map = feature_map.to(torch.float32)

        image_hw = getattr(extractor, "last_image_hw", None)
        if image_hw is None:
            image_hw = (QUERY_DINO_INPUT_SIZE, QUERY_DINO_INPUT_SIZE)
        image_h, image_w = map(int, image_hw)
        expected_hw = (QUERY_DINO_INPUT_SIZE, QUERY_DINO_INPUT_SIZE)
        if (image_h, image_w) != expected_hw:
            raise RuntimeError(
                f"query DINO consumed {(image_h, image_w)}, expected {expected_hw}"
            )
        expected_grid = (QUERY_DINO_PATCH_GRID, QUERY_DINO_PATCH_GRID)
        if tuple(map(int, feature_map.shape[1:])) != expected_grid:
            raise RuntimeError(
                f"query DINO feature grid is {tuple(feature_map.shape[1:])}, "
                f"expected {expected_grid}"
            )

        if feature_dim is None:
            feature_dim = int(feature_map.shape[0])
            device = feature_map.device
            query_points_t = torch.as_tensor(
                query_points,
                dtype=torch.float32,
                device=device,
            )
            view_feature_sum = torch.zeros(
                (len(query_points), feature_dim),
                dtype=torch.float32,
                device=device,
            )
            pixel_feature_sum = torch.zeros_like(view_feature_sum)
            view_counts_t = torch.zeros(
                len(query_points), dtype=torch.long, device=device
            )
            pixel_counts_t = torch.zeros(
                len(query_points), dtype=torch.long, device=device
            )
        else:
            if int(feature_map.shape[0]) != feature_dim:
                raise ValueError("all DINO views must have the same channel count")
            if feature_map.device != query_points_t.device:
                raise ValueError("all DINO views must be encoded on the same device")

        # DINO sees the 224x224 object crop, but geometry remains in the
        # original renderer frame exactly as described by FreeZe.
        depth = np.asarray(template.depth, dtype=np.float64)
        mask = np.asarray(template.mask, dtype=bool)
        foreground = mask & np.isfinite(depth) & (depth > 0)
        pixel_y, pixel_x = np.nonzero(foreground)
        if len(pixel_x) == 0:
            del feature_map
            continue

        per_view_sum = torch.zeros(
            (len(query_points), feature_dim),
            dtype=torch.float32,
            device=query_points_t.device,
        )
        per_view_support = torch.zeros(
            len(query_points), dtype=torch.long, device=query_points_t.device
        )

        for start in range(0, len(pixel_x), pixel_chunk_size):
            stop = min(start + pixel_chunk_size, len(pixel_x))
            pixels = np.stack(
                (pixel_x[start:stop], pixel_y[start:stop]), axis=1
            ).astype(np.int64, copy=False)
            image_xy = renderer_pixels_to_query_crop_xy(
                pixels,
                crop_bbox,
                output_size=QUERY_DINO_INPUT_SIZE,
            ).astype(np.float32, copy=False)
            sampled = sample_feature_map(
                feature_map,
                image_xy,
                image_hw=(image_h, image_w),
            ).to(dtype=torch.float32)

            depths = depth[pixels[:, 1], pixels[:, 0]]
            model_points = rendered_pixels_to_model_points(
                pixels,
                depths,
                template.camera,
            )
            query_ids = _nearest_query_indices_torch(model_points, query_points_t)
            per_view_sum.index_add_(0, query_ids, sampled)
            per_view_support += torch.bincount(
                query_ids,
                minlength=len(query_points),
            )

        visible = per_view_support > 0
        visible_ids = torch.nonzero(visible, as_tuple=False).squeeze(1)
        if len(visible_ids):
            support = per_view_support[visible_ids].to(torch.float32)
            per_view_mean = per_view_sum[visible_ids] / support[:, None]
            view_feature_sum[visible_ids] += per_view_mean
            pixel_feature_sum[visible_ids] += per_view_sum[visible_ids]
            view_counts_t[visible_ids] += 1
            pixel_counts_t += per_view_support

        del feature_map, per_view_sum, per_view_support

    assert feature_dim is not None
    assert view_feature_sum is not None
    return finalize_pixel_lifted_visual_aggregation(
        query_points=query_points,
        view_feature_sum=view_feature_sum.detach().cpu().numpy(),
        pixel_feature_sum=pixel_feature_sum.detach().cpu().numpy(),
        view_counts=view_counts_t.detach().cpu().numpy(),
        pixel_counts=pixel_counts_t.detach().cpu().numpy(),
        min_views=min_views,
    )


def aggregate_query_visual_features_streaming(
    query_points: np.ndarray,
    templates: Sequence[Template],
    extractor,
    depth_tolerance: float,
    min_views: int = 18,
    view_weights=None,
    depth_sampling: str = "inverse_bilinear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy query->image projection aggregator kept for controlled A/B tests.

    Visibility uses the BOP/VisPy pixel-center convention implemented in
    :func:`visible_query_image_coordinates`: rendered depth is sampled at
    ``projected_uv - 0.5``. DINO features are sampled at the original continuous
    projected coordinates, matching FoundPose-style ``grid_sample`` with
    ``align_corners=False``.

    This was the initial reproduction path. The published FreeZe method instead
    describes pixel-level visual features being backprojected into 3D before
    nearest-neighbour association with ``P_Q``; use
    :func:`aggregate_query_visual_features_pixel_lifting_streaming` for that
    paper-style direction.
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
