import numpy as np
import pytest


torch = pytest.importorskip("torch")

from freezev2.onboard import CameraPose, Template
from freezev2.query_features import (
    aggregate_query_visual_features_pixel_lifting_streaming,
    aggregate_query_visual_features_streaming,
)


def test_streaming_query_features_sample_continuous_projection_coordinates():
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
    feature_map = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

    class FakeExtractor:
        last_image_hw = (4, 4)

        def encode(self, image):
            return feature_map

    points, features, counts = aggregate_query_visual_features_streaming(
        query_points,
        [template],
        FakeExtractor(),
        depth_tolerance=1e-8,
        min_views=1,
        depth_sampling="nearest",
    )

    np.testing.assert_allclose(points, query_points)
    np.testing.assert_allclose(features, [[5.0]], atol=1e-6)
    assert counts.tolist() == [1]


def test_pixel_lifted_query_features_aggregate_per_view_and_pixel_support():
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=2,
        direction=np.array([0.0, 0.0, 1.0]),
    )

    depth_a = np.ones((2, 2), dtype=np.float32)
    mask_a = np.array([[True, True], [True, False]])
    depth_a[~mask_a] = 0.0
    depth_b = np.zeros((2, 2), dtype=np.float32)
    depth_b[0, 0] = 1.0

    templates = [
        Template(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=depth_a,
            mask=mask_a,
            camera=camera,
        ),
        Template(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=depth_b,
            mask=depth_b > 0,
            camera=camera,
        ),
    ]
    query_points = np.array([[0.5, 0.5, 1.0]], dtype=np.float64)
    feature_maps = [
        torch.tensor([[[1.0, 3.0], [5.0, 7.0]]]),
        torch.tensor([[[9.0, 11.0], [13.0, 15.0]]]),
    ]

    class FakeExtractor:
        last_image_hw = (2, 2)

        def __init__(self):
            self.index = 0

        def encode(self, image):
            output = feature_maps[self.index]
            self.index += 1
            return output

    (
        points,
        view_uniform,
        pixel_support,
        view_counts,
        pixel_counts,
    ) = aggregate_query_visual_features_pixel_lifting_streaming(
        query_points,
        templates,
        FakeExtractor(),
        min_views=2,
        pixel_chunk_size=2,
    )

    np.testing.assert_allclose(points, query_points)
    # View A mean = (1 + 3 + 5) / 3 = 3; view B mean = 9.
    np.testing.assert_allclose(view_uniform, [[6.0]], atol=1e-6)
    # Pixel-support weighting equals the mean over all four supporting pixels.
    np.testing.assert_allclose(pixel_support, [[4.5]], atol=1e-6)
    np.testing.assert_array_equal(view_counts, [2])
    np.testing.assert_array_equal(pixel_counts, [4])
