from __future__ import annotations

from typing import Sequence

import numpy as np

from .features import sample_feature_map
from .onboard import Template
from .query_features import _view_weight_matrix
from .raster import visible_query_image_coordinates


def _normalize_layers(extractor, layers: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(sorted({int(layer) for layer in layers}))
    if not layers:
        raise ValueError("layers must not be empty")
    block_count = len(extractor.model.blocks)
    if layers[0] < 0 or layers[-1] >= block_count:
        raise ValueError(f"layers must lie inside [0, {block_count - 1}]")
    return layers


def encode_dino_layers(extractor, image, layers: Sequence[int]):
    """Return several normalized DINO patch maps from one backbone forward.

    This mirrors :meth:`DinoExtractor.encode`, but hooks all requested blocks at
    once. It is intended for resolving the paper-omitted ViT-g intermediate
    layer without paying one full 162-view pass per candidate.
    """
    import torch

    layers = _normalize_layers(extractor, layers)
    batch = extractor._prepare_image(image)
    captured = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output

        return hook

    for layer in layers:
        handles.append(
            extractor.model.blocks[layer].register_forward_hook(make_hook(layer))
        )

    try:
        with torch.inference_mode():
            extractor.model((batch - extractor.mean) / extractor.std)
    finally:
        for handle in handles:
            handle.remove()

    if set(captured) != set(layers):
        raise RuntimeError("DINO multi-layer hooks did not capture every requested layer")

    image_h, image_w = batch.shape[-2:]
    patch_h = 1 + (image_h - extractor.patch_size[0]) // extractor.stride[0]
    patch_w = 1 + (image_w - extractor.patch_size[1]) // extractor.stride[1]
    register_tokens = int(getattr(extractor.model, "num_register_tokens", 0))

    feature_maps = {}
    for layer in layers:
        tokens = captured[layer]
        if not torch.is_tensor(tokens) or tokens.ndim != 3:
            raise RuntimeError("DINO token output must have shape BxTxD")
        if hasattr(extractor.model, "norm"):
            tokens = extractor.model.norm(tokens)

        patch_tokens = tokens[:, 1 + register_tokens :, :]
        if patch_tokens.shape[1] != patch_h * patch_w:
            raise RuntimeError(
                "DINO patch-token count does not match the expected feature grid"
            )
        feature_maps[layer] = patch_tokens.reshape(
            patch_tokens.shape[0],
            patch_h,
            patch_w,
            patch_tokens.shape[-1],
        ).permute(0, 3, 1, 2)[0]

    return feature_maps


def aggregate_query_visual_features_multilayer_streaming(
    query_points: np.ndarray,
    templates: Sequence[Template],
    extractor,
    layers: Sequence[int],
    depth_tolerance: float,
    min_views: int = 18,
    view_weights=None,
    depth_sampling: str = "inverse_bilinear",
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Aggregate several DINO intermediate layers in one 162-view pass."""
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if min_views <= 0:
        raise ValueError("min_views must be positive")
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")

    layers = _normalize_layers(extractor, layers)
    if len(templates) == 0:
        return (
            query_points[:0],
            {layer: np.empty((0, 0), dtype=np.float32) for layer in layers},
            np.empty((0,), dtype=np.int32),
        )

    weights = _view_weight_matrix(view_weights, len(templates), len(query_points))
    sums = None
    weight_sums = np.zeros(len(query_points), dtype=np.float64)
    counts = np.zeros(len(query_points), dtype=np.int32)

    for view_id, template in enumerate(templates):
        feature_maps = encode_dino_layers(extractor, template.rgb, layers)
        if sums is None:
            sums = {
                layer: np.zeros(
                    (len(query_points), int(feature_maps[layer].shape[0])),
                    dtype=np.float64,
                )
                for layer in layers
            }

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
        if len(ids):
            inside = (
                (image_xy[:, 0] >= 0.0)
                & (image_xy[:, 0] < image_w)
                & (image_xy[:, 1] >= 0.0)
                & (image_xy[:, 1] < image_h)
            )
            ids = ids[inside]
            image_xy = image_xy[inside]

        if len(ids):
            point_weights = weights[view_id, ids]
            for layer in layers:
                sampled = sample_feature_map(
                    feature_maps[layer],
                    image_xy,
                    image_hw=(image_h, image_w),
                )
                sampled_np = (
                    sampled.detach().to("cpu").numpy().astype(np.float64, copy=False)
                )
                sums[layer][ids] += sampled_np * point_weights[:, None]
            weight_sums[ids] += point_weights
            counts[ids] += 1

        del feature_maps

    assert sums is not None
    keep = (counts >= int(min_views)) & (weight_sums > 0)
    descriptors = {
        layer: (sums[layer][keep] / weight_sums[keep, None]).astype(np.float32)
        for layer in layers
    }
    return query_points[keep].copy(), descriptors, counts[keep].copy()
