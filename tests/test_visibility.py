import numpy as np

from freezev2.onboard import CameraPose, Template
from freezev2.visibility import query_visibility_counts


def test_visibility_counts_respect_depth_and_feature_crop():
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=5,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    depth[4, 4] = 1.0
    template = Template(
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=depth,
        mask=depth > 0,
        camera=camera,
    )
    query_points = np.array([[2.0, 2.0, 1.0], [4.0, 4.0, 1.0]])

    counts = query_visibility_counts(
        query_points,
        [template],
        depth_tolerance=1e-6,
        feature_image_hws=[(4, 4)],
    )

    assert counts.tolist() == [1, 0]


def test_visibility_counts_accumulate_across_views():
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=4,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = 2.0
    template = Template(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        mask=depth > 0,
        camera=camera,
    )
    query_points = np.array([[2.0, 2.0, 2.0]])

    counts = query_visibility_counts(
        query_points,
        [template, template],
        depth_tolerance=1e-6,
    )

    assert counts.tolist() == [2]
