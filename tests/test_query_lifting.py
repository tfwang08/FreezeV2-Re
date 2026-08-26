import numpy as np

from freezev2.onboard import CameraPose
from freezev2.query_features import (
    finalize_pixel_lifted_visual_aggregation,
    query_object_crop_bbox,
    rendered_pixels_to_model_points,
    renderer_pixels_to_query_crop_xy,
)


def test_query_object_crop_uses_tight_mask_bbox_and_224_coordinates():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:9] = True

    bbox = query_object_crop_bbox(mask)
    np.testing.assert_array_equal(bbox, [3, 2, 9, 6])

    pixels = np.array([[3, 2], [8, 5], [5, 3]], dtype=np.int64)
    crop_xy = renderer_pixels_to_query_crop_xy(
        pixels,
        bbox,
        output_size=224,
    )
    expected = np.array(
        [
            [0.5 * 224.0 / 6.0, 0.5 * 224.0 / 4.0],
            [5.5 * 224.0 / 6.0, 3.5 * 224.0 / 4.0],
            [2.5 * 224.0 / 6.0, 1.5 * 224.0 / 4.0],
        ]
    )
    np.testing.assert_allclose(crop_xy, expected, atol=1e-12)


def test_rendered_pixels_backproject_from_half_integer_image_centers():
    camera = CameraPose(
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
        size=2,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    pixels = np.array([[0, 0], [1, 0]], dtype=np.int64)
    depths = np.array([1.0, 1.0], dtype=np.float64)

    points = rendered_pixels_to_model_points(pixels, depths, camera)

    np.testing.assert_allclose(
        points,
        [[0.5, 0.5, 1.0], [1.5, 0.5, 1.0]],
        atol=1e-12,
    )


def test_finalize_pixel_lifted_visual_aggregation_keeps_both_weightings():
    query_points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    # Two view-level means: 3 and 9 -> uniform view mean 6.
    view_feature_sum = np.array([[12.0]], dtype=np.float64)
    # Four supporting pixels with features 1, 3, 5, 9 -> pixel mean 4.5.
    pixel_feature_sum = np.array([[18.0]], dtype=np.float64)
    view_counts = np.array([2], dtype=np.int32)
    pixel_counts = np.array([4], dtype=np.int64)

    points, view_uniform, pixel_support, kept_views, kept_pixels = (
        finalize_pixel_lifted_visual_aggregation(
            query_points=query_points,
            view_feature_sum=view_feature_sum,
            pixel_feature_sum=pixel_feature_sum,
            view_counts=view_counts,
            pixel_counts=pixel_counts,
            min_views=2,
        )
    )

    np.testing.assert_allclose(points, query_points)
    np.testing.assert_allclose(view_uniform, [[6.0]])
    np.testing.assert_allclose(pixel_support, [[4.5]])
    np.testing.assert_array_equal(kept_views, [2])
    np.testing.assert_array_equal(kept_pixels, [4])
