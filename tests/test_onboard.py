import numpy as np
import pytest

from freezev2.onboard import (
    fit_camera_to_mesh,
    make_template_cameras,
    map_visible_pixels_to_query_points,
    render_templates,
    sample_query_points,
    save_onboarding_cache,
)
from freezev2.geometry import project_points


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)


def cube_mesh():
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
    ])
    return Mesh(vertices, faces)


def test_query_sampling_is_exact_deterministic_and_on_surface():
    mesh = cube_mesh()
    a = sample_query_points(mesh, n=5000, seed=0)
    b = sample_query_points(mesh, n=5000, seed=0)
    assert a.shape == (5000, 3)
    assert np.allclose(a, b)
    assert np.all(np.max(np.abs(a), axis=1) <= 1.0 + 1e-12)
    assert np.all(np.isclose(np.max(np.abs(a), axis=1), 1.0, atol=1e-10))


def test_query_sampling_enforces_poisson_disk_spacing():
    points = sample_query_points(cube_mesh(), n=300, seed=0)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    assert float(distance.min()) > 0.15


def test_template_camera_count_rotation_and_cnos_anchors():
    cams = make_template_cameras(n=162)
    assert len(cams) == 162
    for cam in cams:
        assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-7)
        assert np.linalg.det(cam.R) > 0.999999

    directions = np.stack([cam.direction for cam in cams])
    # Anchors decoded from CNOS cam_poses_level1.npy. Ordering is intentionally
    # not asserted because Blender float noise perturbs ties within latitude rings.
    anchors = np.array([
        [0.0, 0.0, -1.0],
        [-0.084441692, -0.259889215, -0.961939275],
        [0.447209895, 0.0, -0.894429028],
    ])
    for anchor in anchors:
        assert np.min(np.linalg.norm(directions - anchor, axis=1)) < 1e-5


def test_fitted_camera_places_cube_at_half_frame():
    mesh = cube_mesh()
    cam = fit_camera_to_mesh(mesh, make_template_cameras(162, size=480)[0], target_fill=0.5)
    uv, z = project_points(mesh.vertices, cam.K, cam.R, cam.t)
    assert np.all(z > 0)
    np.testing.assert_allclose(max(np.ptp(uv, axis=0)), 240.0, atol=0.5)


def test_visibility_mapping_uses_rendered_depth_consistency():
    mesh = cube_mesh()
    cam = fit_camera_to_mesh(mesh, make_template_cameras(162, size=64)[0], target_fill=0.5)
    query = np.array([
        [-0.5, -0.5, -1.0], [0.5, -0.5, -1.0],
        [0.5, 0.5, -1.0], [-0.5, 0.5, -1.0],
    ])
    uv, z = project_points(query, cam.K, cam.R, cam.t)
    pixels = np.rint(uv).astype(int)
    depth = np.zeros((64, 64), dtype=np.float64)
    for (u, v), zz in zip(pixels, z):
        depth[v, u] = zz
    ids, visible_pixels = map_visible_pixels_to_query_points(
        query, depth, cam, depth_tolerance=1e-8
    )
    assert ids.tolist() == [0, 1, 2, 3]
    assert np.array_equal(visible_pixels, pixels)


def test_onboarding_cache_round_trip(tmp_path):
    query = sample_query_points(cube_mesh(), n=8, seed=1)
    cameras = make_template_cameras(162)[:3]
    path = save_onboarding_cache(tmp_path / "obj.npz", query, cameras, diameter=np.array(3.0))
    data = np.load(path)
    assert data["query_points"].shape == (8, 3)
    assert data["camera_R"].shape == (3, 3, 3)
    assert data["camera_t"].shape == (3, 3)
    assert data["camera_K"].shape == (3, 3, 3)
    assert float(data["diameter"]) == 3.0


def test_rendered_cube_depth_maps_back_to_visible_surface():
    pytest.importorskip("pyrender")
    mesh = cube_mesh()
    template = render_templates(
        mesh,
        make_template_cameras(162, size=64)[:1],
        size=64,
        target_fill=0.5,
    )[0]
    assert template.rgb.shape == (64, 64, 3)
    assert template.depth.shape == (64, 64)
    assert template.mask.any()

    query = np.array([[0.0, 0.0, -1.0]])
    ids, _ = map_visible_pixels_to_query_points(
        query, template.depth, template.camera, depth_tolerance=1e-3
    )
    assert ids.tolist() == [0]
