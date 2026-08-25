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
        sampling="nearest",
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
    # Projects to window/image coordinate (1.5, 1.5), i.e. depth[1, 1].
    query_points = np.array([[3.0, 3.0, 2.0]])

    counts = query_visibility_counts(
        query_points,
        [template, template],
        depth_tolerance=1e-6,
        sampling="nearest",
    )

    assert counts.tolist() == [2]


def test_visibility_counts_support_perspective_correct_subpixel_depth():
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=4,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    y, x = np.mgrid[:4, :4]
    inverse_depth = 1.0 + 0.2 * x + 0.1 * y
    depth = (1.0 / inverse_depth).astype(np.float32)
    template = Template(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=depth,
        mask=depth > 0,
        camera=camera,
    )

    raster_u, raster_v = 1.25, 1.5
    image_u, image_v = raster_u + 0.5, raster_v + 0.5
    z = 1.0 / (1.0 + 0.2 * raster_u + 0.1 * raster_v)
    query_points = np.array([[image_u * z, image_v * z, z]])

    counts = query_visibility_counts(
        query_points,
        [template],
        depth_tolerance=1e-6,
        sampling="inverse_bilinear",
    )

    assert counts.tolist() == [1]
