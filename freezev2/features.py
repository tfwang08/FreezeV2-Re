from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .onboard import Template, map_visible_pixels_to_query_points


DINOV2_FOUNDPOSE_COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
DINOV2_HUB_REPO = f"facebookresearch/dinov2:{DINOV2_FOUNDPOSE_COMMIT}"
DINOV2_MODEL_NAME = "dinov2_vitg14"


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install feature dependencies with: pip install -e '.[features]'"
        ) from exc
    return torch


def sample_feature_map(feature_map_chw, pixels_xy, image_hw: tuple[int, int]):
    """Bilinearly sample a CxHxW feature map at image-space (x, y) points.

    This follows FoundPose exactly: image coordinates are mapped to [-1, 1]
    with ``2 * point / (image_width, image_height) - 1`` and sampled with
    ``grid_sample(..., align_corners=False)``.
    """
    torch = _torch()
    import torch.nn.functional as F

    feature_map = torch.as_tensor(feature_map_chw)
    if feature_map.ndim != 3:
        raise ValueError("feature_map_chw must have shape CxHxW")
    if not feature_map.is_floating_point():
        feature_map = feature_map.to(torch.float32)

    points = torch.as_tensor(
        pixels_xy,
        dtype=feature_map.dtype,
        device=feature_map.device,
    )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixels_xy must have shape Nx2")

    image_h, image_w = map(int, image_hw)
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_hw must be positive")
    if len(points) == 0:
        return feature_map.new_empty((0, feature_map.shape[0]))

    image_size = torch.tensor(
        [image_w, image_h],
        dtype=points.dtype,
        device=points.device,
    )
    uv = (2.0 / image_size) * points - 1.0
    query_coords = uv.unsqueeze(0).unsqueeze(2)
    sampled = F.grid_sample(
        feature_map.unsqueeze(0),
        query_coords,
        align_corners=False,
    )
    return sampled[0, :, :, 0].permute(1, 0)


def _pair(value) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("expected a scalar or pair")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class DinoExtractor:
    """Frozen DINOv2 intermediate token extractor for FreeZeV2.

    The paper specifies ViT-g/14 intermediate patch features but does not
    publish the giant-backbone layer index. ``layer`` is therefore mandatory
    and remains part of the experiment configuration rather than a hidden
    default.
    """

    def __init__(
        self,
        device: str,
        layer: int,
        facet: str = "token",
        model_name: str = DINOV2_MODEL_NAME,
        model=None,
        repo_or_dir: str | Path | None = None,
    ) -> None:
        torch = _torch()
        if facet != "token":
            raise ValueError("Task 3 currently supports facet='token' only")

        self.device = torch.device(device)
        self.layer = int(layer)
        self.facet = facet
        self.model_name = model_name

        if model is None:
            if repo_or_dir is None:
                model = torch.hub.load(
                    DINOV2_HUB_REPO,
                    model_name,
                    pretrained=True,
                    trust_repo=True,
                )
            else:
                model = torch.hub.load(
                    str(repo_or_dir),
                    model_name,
                    pretrained=True,
                    source="local",
                )

        self.model = model.to(self.device)
        if not hasattr(self.model, "blocks"):
            raise ValueError("DINO model must expose transformer blocks")
        if self.layer < 0 or self.layer >= len(self.model.blocks):
            raise ValueError(
                f"layer {self.layer} is outside [0, {len(self.model.blocks) - 1}]"
            )
        if not hasattr(self.model, "patch_embed"):
            raise ValueError("DINO model must expose patch_embed")

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.patch_size = _pair(self.model.patch_embed.patch_size)
        self.stride = _pair(self.model.patch_embed.proj.stride)
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

    def _prepare_image(self, image):
        torch = _torch()
        tensor = torch.as_tensor(image, device=self.device)
        if tensor.ndim != 3:
            raise ValueError("image must have shape HxWx3 or 3xHxW")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError("image must have exactly 3 channels")

        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32) / 255.0
        else:
            tensor = tensor.to(torch.float32)
            if tensor.numel() and (
                float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0
            ):
                raise ValueError("floating-point image values must be in [0, 1]")
        return tensor.unsqueeze(0)

    def encode(self, image):
        """Return one intermediate DINO patch feature map as CxHf xWf."""
        torch = _torch()
        batch = self._prepare_image(image)
        captured = []

        def hook(_module, _inputs, output):
            captured.append(output)

        handle = self.model.blocks[self.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.model((batch - self.mean) / self.std)
        finally:
            handle.remove()

        if len(captured) != 1 or not torch.is_tensor(captured[0]):
            raise RuntimeError("DINO token hook did not return one tensor")
        tokens = captured[0]
        if tokens.ndim != 3:
            raise RuntimeError("DINO token output must have shape BxTxD")

        if hasattr(self.model, "norm"):
            tokens = self.model.norm(tokens)

        register_tokens = int(getattr(self.model, "num_register_tokens", 0))
        patch_tokens = tokens[:, 1 + register_tokens :, :]
        image_h, image_w = batch.shape[-2:]
        patch_h = 1 + (image_h - self.patch_size[0]) // self.stride[0]
        patch_w = 1 + (image_w - self.patch_size[1]) // self.stride[1]
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("image is smaller than the DINO patch size")
        if patch_tokens.shape[1] != patch_h * patch_w:
            raise RuntimeError(
                "DINO patch-token count does not match the expected feature grid"
            )

        feature_map = patch_tokens.reshape(
            patch_tokens.shape[0], patch_h, patch_w, patch_tokens.shape[-1]
        ).permute(0, 3, 1, 2)
        return feature_map[0]


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


def aggregate_query_visual_features(
    query_points: np.ndarray,
    templates: Sequence[Template],
    feature_maps: Sequence,
    depth_tolerance: float,
    min_views: int = 18,
    view_weights=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate rendered DINO observations onto query surface points.

    FreeZeV2 keeps points visible in at least 18 views and describes the visual
    descriptor as a weighted average of per-view DINO features. The paper does
    not publish the weighting rule, so weights remain explicit. ``None`` means
    uniform averaging and is intended for smoke tests until that missing detail
    is resolved. Raw visual means are returned here; Eq. (1) applies PCA and
    L2-normalization later during visual/geometric fusion.
    """
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape Nx3")
    if len(templates) != len(feature_maps):
        raise ValueError("templates and feature_maps must have the same length")
    if min_views <= 0:
        raise ValueError("min_views must be positive")
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")
    if len(feature_maps) == 0:
        return (
            query_points[:0],
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    feature_dim = int(feature_maps[0].shape[0])
    for feature_map in feature_maps:
        if getattr(feature_map, "ndim", None) != 3:
            raise ValueError("each feature map must have shape CxHxW")
        if int(feature_map.shape[0]) != feature_dim:
            raise ValueError("all feature maps must have the same channel count")

    weights = _view_weight_matrix(view_weights, len(templates), len(query_points))
    sums = np.zeros((len(query_points), feature_dim), dtype=np.float64)
    weight_sums = np.zeros(len(query_points), dtype=np.float64)
    counts = np.zeros(len(query_points), dtype=np.int32)

    for view_id, (template, feature_map) in enumerate(zip(templates, feature_maps)):
        ids, pixels = map_visible_pixels_to_query_points(
            query_points,
            template.depth,
            template.camera,
            depth_tolerance=depth_tolerance,
        )
        if len(ids) == 0:
            continue
        sampled = sample_feature_map(
            feature_map,
            pixels,
            image_hw=template.depth.shape[:2],
        )
        sampled_np = sampled.detach().to("cpu").numpy().astype(np.float64, copy=False)
        point_weights = weights[view_id, ids]
        sums[ids] += sampled_np * point_weights[:, None]
        weight_sums[ids] += point_weights
        counts[ids] += 1

    keep = (counts >= int(min_views)) & (weight_sums > 0)
    descriptors = sums[keep] / weight_sums[keep, None]
    return (
        query_points[keep].copy(),
        descriptors.astype(np.float32),
        counts[keep].copy(),
    )
