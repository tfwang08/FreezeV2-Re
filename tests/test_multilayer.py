import numpy as np
import pytest


torch = pytest.importorskip("torch")
import torch.nn as nn

from freezev2.features import DinoExtractor
from freezev2.multilayer import (
    aggregate_query_visual_features_multilayer_streaming,
    encode_dino_layers,
)
from freezev2.onboard import CameraPose, Template


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
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        patch_h, patch_w = self.patch_embed.patch_size
        h, w = x.shape[-2] // patch_h, x.shape[-1] // patch_w
        patch_tokens = torch.zeros((x.shape[0], h * w, 3), device=x.device)
        cls_token = torch.zeros((x.shape[0], 1, 3), device=x.device)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


def test_encode_dino_layers_uses_one_forward_for_multiple_layers():
    model = FakeDino()
    extractor = DinoExtractor(device="cpu", layer=0, model=model)

    maps = encode_dino_layers(
        extractor,
        np.zeros((4, 4, 3), dtype=np.uint8),
        layers=[0, 1],
    )

    assert model.forward_calls == 1
    assert list(maps) == [0, 1]
    assert maps[0].shape == (3, 2, 2)
    assert torch.allclose(maps[0], torch.ones_like(maps[0]))
    assert torch.allclose(maps[1], 2.0 * torch.ones_like(maps[1]))


def test_multilayer_streaming_aggregates_all_layers_from_one_view_forward():
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=4,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = 1.0
    template = Template(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        mask=depth > 0,
        camera=camera,
    )
    query_points = np.array([[1.5, 1.5, 1.0]])

    model = FakeDino()
    extractor = DinoExtractor(device="cpu", layer=0, model=model)

    points, features_by_layer, counts = (
        aggregate_query_visual_features_multilayer_streaming(
            query_points,
            [template],
            extractor,
            layers=[0, 1],
            depth_tolerance=1e-8,
            min_views=1,
            depth_sampling="nearest",
        )
    )

    assert model.forward_calls == 1
    np.testing.assert_allclose(points, query_points)
    assert counts.tolist() == [1]
    np.testing.assert_allclose(features_by_layer[0], np.ones((1, 3)), atol=1e-6)
    np.testing.assert_allclose(features_by_layer[1], 2.0 * np.ones((1, 3)), atol=1e-6)
