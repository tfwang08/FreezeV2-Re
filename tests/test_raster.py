import numpy as np

from freezev2.onboard import CameraPose
from freezev2.raster import visible_query_image_coordinates


def _camera(size=4):
    return CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=size,
        direction=np.array([0.0, 0.0, 1.0]),
    )


def test_render_visibility_uses_opengl_half_pixel_centers():
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = 1.0
    query_points = np.array([[1.5, 1.5, 1.0]])

    ids, image_xy = visible_query_image_coordinates(
        query_points,
        depth,
        _camera(),
        depth_tolerance=1e-8,
        sampling="nearest",
    )

    assert ids.tolist() == [0]
    np.testing.assert_allclose(image_xy, [[1.5, 1.5]])


def test_inverse_bilinear_matches_perspective_planar_depth():
    y, x = np.mgrid[:4, :4]
    inverse_depth = 1.0 + 0.2 * x + 0.1 * y
    depth = (1.0 / inverse_depth).astype(np.float32)

    raster_u, raster_v = 1.25, 1.5
    image_u, image_v = raster_u + 0.5, raster_v + 0.5
    z = 1.0 / (1.0 + 0.2 * raster_u + 0.1 * raster_v)
    query_points = np.array([[image_u * z, image_v * z, z]])

    ids, image_xy = visible_query_image_coordinates(
        query_points,
        depth,
        _camera(),
        depth_tolerance=1e-6,
        sampling="inverse_bilinear",
    )

    assert ids.tolist() == [0]
    np.testing.assert_allclose(image_xy, [[image_u, image_v]], atol=1e-12)
