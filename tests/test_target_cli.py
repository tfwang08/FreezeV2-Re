import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


_RUN_BOP_PATH = Path(__file__).resolve().parents[1] / "run_bop.py"
_RUN_BOP_SPEC = importlib.util.spec_from_file_location("run_bop", _RUN_BOP_PATH)
assert _RUN_BOP_SPEC is not None and _RUN_BOP_SPEC.loader is not None
run_bop = importlib.util.module_from_spec(_RUN_BOP_SPEC)
_RUN_BOP_SPEC.loader.exec_module(run_bop)


def test_mask_patch_centers_use_smallest_square_bbox():
    assert hasattr(run_bop, "_sample_mask_patch_centers")

    mask = np.zeros((80, 100), dtype=bool)
    mask[30:50, 20:62] = True  # 42 px wide, 20 px tall.
    centers, pixels = run_bop._sample_mask_patch_centers(mask, grid_size=4)

    # The 42x20 rectangle is expanded about its center to a 42x42 square.
    # Four grid rows are centered at y={24.25,34.75,45.25,55.75}; only the
    # middle two rows have centers inside the original mask.
    assert centers.shape == (8, 2)
    assert pixels.shape == (8, 2)
    np.testing.assert_allclose(
        np.unique(centers[:, 0]),
        [25.25, 35.75, 46.25, 56.75],
    )
    np.testing.assert_allclose(np.unique(centers[:, 1]), [34.75, 45.25])
    np.testing.assert_array_equal(pixels, np.floor(centers).astype(np.int64))
    assert np.all(mask[pixels[:, 1], pixels[:, 0]])


def test_target_patch_grid_preserves_direct_dino_token_indices():
    assert hasattr(run_bop, "_target_patch_grid")

    mask = np.ones((64, 64), dtype=bool)
    centers, pixels, token_indices, bbox = run_bop._target_patch_grid(
        mask,
        grid_size=16,
    )

    assert centers.shape == (256, 2)
    assert pixels.shape == (256, 2)
    np.testing.assert_array_equal(token_indices, np.arange(256, dtype=np.int64))
    np.testing.assert_allclose(bbox, [0.0, 0.0, 64.0, 64.0])
    expected_axis = np.arange(2.0, 64.0, 4.0, dtype=np.float32)
    np.testing.assert_allclose(np.unique(centers[:, 0]), expected_axis)
    np.testing.assert_allclose(np.unique(centers[:, 1]), expected_axis)


def test_extract_target_builds_sparse_dense_and_128d_representation(tmp_path, monkeypatch):
    bop_root = tmp_path / "data" / "bop"
    scene_dir = bop_root / "lmo" / "test" / "000002"
    (scene_dir / "rgb").mkdir(parents=True)
    (scene_dir / "depth").mkdir()

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(64, dtype=np.uint8)[None, :]
    depth_raw = np.full((64, 64), 1000, dtype=np.uint16)
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    Image.fromarray(rgb).save(scene_dir / "rgb" / "000003.png")
    Image.fromarray(depth_raw).save(scene_dir / "depth" / "000003.png")
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    K = [100.0, 0.0, 32.0, 0.0, 100.0, 32.0, 0.0, 0.0, 1.0]
    (scene_dir / "scene_camera.json").write_text(
        json.dumps({"3": {"cam_K": K, "depth_scale": 0.1}})
    )

    query_cache = tmp_path / "query.npz"
    pca_components = np.zeros((64, 96), dtype=np.float32)
    pca_components[:, :64] = np.eye(64, dtype=np.float32)
    np.savez_compressed(
        query_cache,
        query_points=np.zeros((5, 3), dtype=np.float32),
        fused_features=np.zeros((5, 128), dtype=np.float32),
        pca_mean=np.zeros(96, dtype=np.float32),
        pca_components=pca_components,
        pca_dim=np.int32(64),
        diameter=np.float32(100.0),
        dino_layer=np.int32(30),
        dino_facet=np.array("token"),
        dino_model=np.array("dinov2_vitg14_reg"),
    )

    dinov2_root = tmp_path / "dinov2"
    gedi_root = tmp_path / "gedi"
    dinov2_root.mkdir()
    gedi_root.mkdir()
    checkpoint = tmp_path / "gedi.tar"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "target.npz"
    calls = {}

    class FakeDinoExtractor:
        def __init__(self, device, layer, facet="token", model_name="dinov2_vitg14", repo_or_dir=None):
            calls["dino_init"] = (device, layer, facet, model_name, Path(repo_or_dir))
            self.last_image_hw = None

        def encode(self, image):
            crop = np.asarray(image)
            calls["rgb_shape"] = tuple(crop.shape)
            assert crop.shape == (224, 224, 3)
            self.last_image_hw = (224, 224)
            feature_map = np.empty((96, 16, 16), dtype=np.float32)
            token_ids = np.arange(256, dtype=np.float32).reshape(16, 16)
            for channel in range(96):
                feature_map[channel] = token_ids + channel / 1000.0
            return feature_map

    class FakeGediExtractor:
        def __init__(self, checkpoint, gedi_root, seed=0):
            calls["gedi_init"] = (Path(checkpoint), Path(gedi_root), seed)

        def encode(self, pts, cloud, object_diameter):
            calls["gedi_shapes"] = (tuple(pts.shape), tuple(cloud.shape), object_diameter)
            values = np.linspace(1.0, 2.0, 64, dtype=np.float32)
            return np.tile(values, (len(pts), 1))

    def fail_if_interpolated(*_args, **_kwargs):
        raise AssertionError("target DINO descriptors must use direct patch tokens")

    monkeypatch.setattr(run_bop, "DinoExtractor", FakeDinoExtractor, raising=False)
    monkeypatch.setattr(run_bop, "GediExtractor", FakeGediExtractor, raising=False)
    monkeypatch.setattr(run_bop, "sample_feature_map", fail_if_interpolated, raising=False)
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "extract-target",
        "--dataset",
        "lmo",
        "--scene-id",
        "2",
        "--im-id",
        "3",
        "--obj-id",
        "1",
        "--mask",
        str(mask_path),
        "--bop-root",
        str(bop_root),
        "--query-cache",
        str(query_cache),
        "--dinov2-root",
        str(dinov2_root),
        "--gedi-root",
        str(gedi_root),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--seed",
        "7",
    ])

    run_bop.main()

    assert calls["dino_init"] == (
        "cuda",
        30,
        "token",
        "dinov2_vitg14_reg",
        dinov2_root,
    )
    assert calls["rgb_shape"] == (224, 224, 3)
    assert calls["gedi_shapes"] == ((256, 3), (3000, 3), 100.0)

    with np.load(output, allow_pickle=False) as data:
        sparse_pixels = np.asarray(data["sparse_pixels"])
        sparse_image_xy = np.asarray(data["sparse_image_xy"])
        sparse_points = np.asarray(data["sparse_points"])
        dense_points = np.asarray(data["dense_points"])
        visual = np.asarray(data["visual_features"])
        target = np.asarray(data["target_features"])
        assert sparse_pixels.shape == (256, 2)
        assert sparse_image_xy.shape == (256, 2)
        assert sparse_points.shape == (256, 3)
        assert dense_points.shape == (3000, 3)
        assert visual.shape == (256, 96)
        assert data["geometric_features"].shape == (256, 64)
        assert target.shape == (256, 128)
        expected_axis = np.arange(2.0, 64.0, 4.0, dtype=np.float32)
        np.testing.assert_allclose(np.unique(sparse_image_xy[:, 0]), expected_axis)
        np.testing.assert_allclose(np.unique(sparse_image_xy[:, 1]), expected_axis)
        np.testing.assert_array_equal(sparse_pixels, np.floor(sparse_image_xy).astype(np.int32))
        np.testing.assert_allclose(visual[:, 0], np.arange(256, dtype=np.float32))
        np.testing.assert_allclose(sparse_points[:, 2], 100.0)
        np.testing.assert_allclose(dense_points[:, 2], 100.0)
        np.testing.assert_allclose(sparse_points[:, 0], sparse_image_xy[:, 0] - 32.0)
        np.testing.assert_allclose(sparse_points[:, 1], sparse_image_xy[:, 1] - 32.0)
        np.testing.assert_allclose(np.linalg.norm(target[:, :64], axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(target[:, 64:], axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(float(data["depth_scale"]), 0.1, atol=1e-7)
        np.testing.assert_allclose(data["square_bbox_xyxy"], [0.0, 0.0, 64.0, 64.0])
        np.testing.assert_array_equal(data["target_dino_input_hw"], [224, 224])
        np.testing.assert_array_equal(data["target_dino_patch_grid"], [16, 16])
        assert str(np.asarray(data["target_dino_mode"]).item()) == "square_crop_224_direct_tokens"
        assert int(data["grid_size"]) == 16
        assert int(data["dense_size_requested"]) == 3000
        assert np.isfinite(target).all()


def test_visualize_target_writes_overlay_and_reports_geometry(tmp_path, monkeypatch, capsys):
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    output = tmp_path / "overlay.png"
    cache_path = tmp_path / "target.npz"

    rgb = np.full((30, 40, 3), 40, dtype=np.uint8)
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:25, 8:32] = 255
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(mask).save(mask_path)

    image_xy = np.array([[10.5, 8.5], [20.5, 15.5], [30.5, 22.5]], dtype=np.float32)
    points = np.array([
        [-5.0, -2.0, 400.0],
        [0.0, 1.0, 410.0],
        [6.0, 3.0, 420.0],
    ], dtype=np.float32)
    np.savez_compressed(
        cache_path,
        sparse_image_xy=image_xy,
        sparse_points=points,
        rgb_source=np.array(str(rgb_path)),
        mask_source=np.array(str(mask_path)),
    )

    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "visualize-target",
        "--target-cache",
        str(cache_path),
        "--output",
        str(output),
        "--radius",
        "2",
    ])

    run_bop.main()

    assert output.is_file()
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert rendered.shape == (30, 40, 3)
    assert not np.array_equal(rendered, rgb)

    report = json.loads(capsys.readouterr().out)
    assert report["point_count"] == 3
    assert report["inside_mask_count"] == 3
    assert report["inside_mask_fraction"] == 1.0
    np.testing.assert_allclose(report["xyz_min"], [-5.0, -2.0, 400.0])
    np.testing.assert_allclose(report["xyz_max"], [6.0, 3.0, 420.0])
    assert report["output"] == str(output)
