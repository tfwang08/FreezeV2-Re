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


class _FakeSampled:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def numpy(self):
        return self.array


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
        dino_model=np.array("dinov2_vitg14"),
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

        def compatible_image_hw(self, image_hw):
            assert tuple(image_hw) == (64, 64)
            return (56, 56)

        def encode(self, image):
            calls["rgb_shape"] = tuple(np.asarray(image).shape)
            self.last_image_hw = (56, 56)
            return object()

    class FakeGediExtractor:
        def __init__(self, checkpoint, gedi_root, seed=0):
            calls["gedi_init"] = (Path(checkpoint), Path(gedi_root), seed)

        def encode(self, pts, cloud, object_diameter):
            calls["gedi_shapes"] = (tuple(pts.shape), tuple(cloud.shape), object_diameter)
            values = np.linspace(1.0, 2.0, 64, dtype=np.float32)
            return np.tile(values, (len(pts), 1))

    def fake_sample_feature_map(_feature_map, pixels_xy, image_hw):
        pixels_xy = np.asarray(pixels_xy, dtype=np.float32)
        calls["sample_pixels"] = pixels_xy.copy()
        calls["sample_hw"] = tuple(image_hw)
        values = np.linspace(1.0, 2.0, 96, dtype=np.float32)
        return _FakeSampled(np.tile(values, (len(pixels_xy), 1)))

    monkeypatch.setattr(run_bop, "DinoExtractor", FakeDinoExtractor, raising=False)
    monkeypatch.setattr(run_bop, "GediExtractor", FakeGediExtractor, raising=False)
    monkeypatch.setattr(run_bop, "sample_feature_map", fake_sample_feature_map, raising=False)
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
        "dinov2_vitg14",
        dinov2_root,
    )
    assert calls["rgb_shape"] == (64, 64, 3)
    assert calls["sample_hw"] == (56, 56)
    assert calls["gedi_shapes"] == ((256, 3), (3000, 3), 100.0)
    sampled = calls["sample_pixels"]
    assert sampled.shape == (256, 2)
    np.testing.assert_allclose(sampled % 1.0, 0.5)
    assert np.all(sampled[:, 0] < 56.0)
    assert np.all(sampled[:, 1] < 56.0)

    with np.load(output, allow_pickle=False) as data:
        sparse_pixels = np.asarray(data["sparse_pixels"])
        sparse_points = np.asarray(data["sparse_points"])
        dense_points = np.asarray(data["dense_points"])
        target = np.asarray(data["target_features"])
        assert sparse_pixels.shape == (256, 2)
        assert sparse_points.shape == (256, 3)
        assert dense_points.shape == (3000, 3)
        assert data["visual_features"].shape == (256, 96)
        assert data["geometric_features"].shape == (256, 64)
        assert target.shape == (256, 128)
        np.testing.assert_allclose(sparse_points[:, 2], 100.0)
        np.testing.assert_allclose(dense_points[:, 2], 100.0)
        np.testing.assert_allclose(np.linalg.norm(target[:, :64], axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(target[:, 64:], axis=1), 1.0, atol=1e-5)
        assert float(data["depth_scale"]) == 0.1
        assert int(data["grid_size"]) == 16
        assert int(data["dense_size_requested"]) == 3000
        assert np.isfinite(target).all()
