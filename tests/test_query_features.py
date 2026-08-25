import numpy as np
import pytest


torch = pytest.importorskip("torch")

from freezev2.onboard import CameraPose, Template
from freezev2.query_features import aggregate_query_visual_features_streaming


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
