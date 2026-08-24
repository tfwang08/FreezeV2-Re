import numpy as np
import pytest


torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from freezev2.features import (
    DINOV2_FOUNDPOSE_COMMIT,
    DinoExtractor,
    aggregate_query_visual_features,
    sample_feature_map,
)
from freezev2.onboard import CameraPose, Template


def test_sample_feature_map_matches_foundpose_grid_sample_convention():
    feature_map = torch.tensor([[[0.0, 1.0], [10.0, 11.0]]])
    points = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    got = sample_feature_map(feature_map, points, image_hw=(4, 4))

    uv = (2.0 / torch.tensor([4.0, 4.0])) * points - 1.0
    expected = F.grid_sample(
        feature_map.unsqueeze(0),
        uv.unsqueeze(0).unsqueeze(2),
        align_corners=False,
    )[0, :, :, 0].permute(1, 0)
    assert torch.allclose(got, expected)


class AddOne(nn.Module):
    def forward(self, x):
        return x + 1.0


class FakePatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = (2, 2)
        self.proj = nn.Conv2d(3, 3, kernel_size=2, stride=2, bias=False)


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = FakePatchEmbed()
        self.blocks = nn.ModuleList([AddOne(), AddOne()])
        self.norm = nn.Identity()
        self.num_register_tokens = 0
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        h, w = x.shape[-2] // 2, x.shape[-1] // 2
        patch_tokens = torch.zeros((x.shape[0], h * w, 3), device=x.device)
        cls_token = torch.zeros((x.shape[0], 1, 3), device=x.device)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


def test_dino_extractor_is_frozen_and_returns_intermediate_patch_map():
    extractor = DinoExtractor(device="cpu", layer=0, model=FakeDino())
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    feature_map = extractor.encode(image)

    assert feature_map.shape == (3, 2, 2)
    assert extractor.model.training is False
    assert all(not p.requires_grad for p in extractor.model.parameters())
    assert torch.allclose(feature_map, torch.ones_like(feature_map))


def _template(depth_value=2.0):
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=4,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = depth_value
    return Template(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        mask=depth > 0,
        camera=camera,
    )


def test_query_visual_aggregation_filters_visibility_and_supports_weights():
    query_points = np.array([[2.0, 2.0, 2.0]])
    templates = [_template(), _template()]
    feature_a = torch.zeros((2, 4, 4))
    feature_b = torch.zeros((2, 4, 4))
    feature_a[0] = 1.0
    feature_b[1] = 1.0

    points, features, counts = aggregate_query_visual_features(
        query_points,
        templates,
        [feature_a, feature_b],
        depth_tolerance=1e-6,
        min_views=2,
        view_weights=np.array([[1.0], [3.0]]),
    )

    assert points.shape == (1, 3)
    assert counts.tolist() == [2]
    np.testing.assert_allclose(features[0], [0.25, 0.75], atol=1e-6)

    points, features, counts = aggregate_query_visual_features(
        query_points,
        templates,
        [feature_a, feature_b],
        depth_tolerance=1e-6,
        min_views=3,
    )
    assert points.shape == (0, 3)
    assert features.shape == (0, 2)
    assert counts.shape == (0,)


def test_foundpose_dinov2_commit_is_pinned():
    assert DINOV2_FOUNDPOSE_COMMIT == "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
