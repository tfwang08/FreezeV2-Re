import sys
import types

import numpy as np
import pytest

from freezev2.onboard import (
    fit_camera_to_mesh,
    load_onboarding_templates,
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


def test_onboarding_cache_can_reload_rgb_depth_and_cameras(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    query = sample_query_points(cube_mesh(), n=8, seed=2)
    cameras = make_template_cameras(162, size=4)[:2]
    depths = np.zeros((2, 4, 4), dtype=np.float32)
    depths[0, 1:3, 1:3] = 2.0
    depths[1, 1:3, 1:3] = 3.0

    cache = save_onboarding_cache(
        tmp_path / "onboarding.npz",
        query,
        cameras,
        template_depth=depths,
    )
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    Image.fromarray(np.full((4, 4, 3), 10, dtype=np.uint8)).save(rgb_dir / "000.png")
    Image.fromarray(np.full((4, 4, 3), 20, dtype=np.uint8)).save(rgb_dir / "001.png")

    templates = load_onboarding_templates(cache, rgb_dir)

    assert len(templates) == 2
    assert templates[0].rgb[0, 0, 0] == 10
    assert templates[1].rgb[0, 0, 0] == 20
    np.testing.assert_allclose(templates[0].depth, depths[0])
    np.testing.assert_allclose(templates[1].depth, depths[1])
    np.testing.assert_allclose(templates[1].camera.R, cameras[1].R)
    np.testing.assert_allclose(templates[1].camera.t, cameras[1].t)
    assert templates[1].camera.size == 4


def test_render_templates_uses_bop_vispy_backend(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    source = cube_mesh()
    mesh = trimesh.Trimesh(vertices=source.vertices, faces=source.faces, process=False)
    mesh_path = tmp_path / "cube.ply"
    mesh.export(mesh_path)

    calls = {}

    class FakeRenderer:
        def set_light_cam_pos(self, value):
            calls["light_pos"] = tuple(value)

        def set_light_ambient_weight(self, value):
            calls["ambient"] = value

        def add_object(self, obj_id, model_path, **kwargs):
            calls["obj_id"] = obj_id
            calls["model_path"] = str(model_path)
            calls["add_kwargs"] = kwargs

        def render_object(self, obj_id, R, t, fx, fy, cx, cy):
            calls["render"] = (obj_id, R.copy(), t.copy(), fx, fy, cx, cy)
            depth = np.zeros((64, 64), dtype=np.float32)
            depth[16:48, 16:48] = 3.0
            return {
                "rgb": np.full((64, 64, 3), 127, dtype=np.uint8),
                "depth": depth,
            }

        def remove_object(self, obj_id):
            calls["removed"] = obj_id

    renderer_module = types.ModuleType("bop_toolkit_lib.rendering.renderer")

    def create_renderer(width, height, **kwargs):
        calls["create"] = (width, height, kwargs)
        return FakeRenderer()

    renderer_module.create_renderer = create_renderer
    rendering_module = types.ModuleType("bop_toolkit_lib.rendering")
    rendering_module.renderer = renderer_module
    bop_module = types.ModuleType("bop_toolkit_lib")
    bop_module.rendering = rendering_module
    monkeypatch.setitem(sys.modules, "bop_toolkit_lib", bop_module)
    monkeypatch.setitem(sys.modules, "bop_toolkit_lib.rendering", rendering_module)
    monkeypatch.setitem(sys.modules, "bop_toolkit_lib.rendering.renderer", renderer_module)

    template = render_templates(
        mesh_path,
        make_template_cameras(162, size=64)[:1],
        size=64,
        target_fill=0.5,
    )[0]

    assert calls["create"] == (
        64,
        64,
        {
            "renderer_type": "vispy",
            "mode": "rgb+depth",
            "shading": "flat",
            "bg_color": (0.0, 0.0, 0.0, 0.0),
        },
    )
    assert calls["obj_id"] == 1
    assert calls["removed"] == 1
    assert template.rgb.shape == (64, 64, 3)
    assert template.depth.shape == (64, 64)
    assert template.mask.any()


def test_rendered_cube_depth_maps_back_to_visible_surface(tmp_path):
    pytest.importorskip("bop_toolkit_lib")
    trimesh = pytest.importorskip("trimesh")
    source = cube_mesh()
    mesh = trimesh.Trimesh(vertices=source.vertices, faces=source.faces, process=False)
    mesh_path = tmp_path / "cube.ply"
    mesh.export(mesh_path)

    template = render_templates(
        mesh_path,
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
