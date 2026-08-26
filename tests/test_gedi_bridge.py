from pathlib import Path

import numpy as np

import freezev2.gedi_bridge as gedi_bridge
from freezev2.gedi_bridge import (
    GEDI_REPO_COMMIT,
    _load_official_gedi_class,
    build_gedi_config,
    concatenate_gedi_scales,
)
from freezev2.gedi_radius import radius_search


def test_gedi_config_uses_freeze_scales_and_official_inference_shape():
    cfg30 = build_gedi_config(100.0, 0.30, Path("checkpoint.tar"))
    cfg40 = build_gedi_config(100.0, 0.40, Path("checkpoint.tar"))

    assert cfg30["dim"] == 32
    assert cfg30["samples_per_batch"] == 500
    assert cfg30["samples_per_patch_lrf"] == 4000
    assert cfg30["samples_per_patch_out"] == 512
    assert cfg30["r_lrf"] == 30.0
    assert cfg40["r_lrf"] == 40.0


def test_gedi_scale_concatenation_is_64d_and_ordered_30_then_40():
    f30 = np.tile(np.arange(32), (3, 1)).astype(np.float32)
    f40 = np.tile(np.arange(32, 64), (3, 1)).astype(np.float32)

    fused = concatenate_gedi_scales(f30, f40)

    assert fused.shape == (3, 64)
    np.testing.assert_array_equal(fused[:, :32], f30)
    np.testing.assert_array_equal(fused[:, 32:], f40)


def test_official_gedi_revision_is_pinned():
    assert GEDI_REPO_COMMIT == "b3dd86776750d8221f89d39975118da9839b39f7"


def test_official_gedi_loader_injects_radius_search_shim(tmp_path):
    (tmp_path / "gedi.py").write_text(
        "import open3d.ml.torch as ml3d\n"
        "class GeDi:\n"
        "    radius_search = staticmethod(ml3d.ops.radius_search)\n"
    )

    cls = _load_official_gedi_class(tmp_path)

    assert cls.radius_search is radius_search


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def float(self):
        return self


class _FakeCuda:
    def is_available(self):
        return True

    def manual_seed_all(self, seed):
        pass

    def empty_cache(self):
        pass


class _FakeTorch:
    cuda = _FakeCuda()

    def manual_seed(self, seed):
        pass

    def from_numpy(self, array):
        return _FakeTensor(array)


class _FakeGeDi:
    configs = []

    def __init__(self, config):
        self.config = config
        self.__class__.configs.append(config)

    def compute(self, pts, pcd):
        assert pcd.array.shape[1] == 3
        return np.full(
            (len(pts.array), 32),
            self.config["r_lrf"],
            dtype=np.float32,
        )


def test_gedi_extractor_runs_two_freeze_scales_and_returns_64d():
    _FakeGeDi.configs.clear()
    points = np.arange(15, dtype=np.float32).reshape(5, 3)

    extractor = gedi_bridge.GediExtractor(
        checkpoint="checkpoint.tar",
        gedi_root="external/gedi",
        backend=(_FakeTorch(), _FakeGeDi),
    )
    features = extractor.encode(points, points, object_diameter=100.0)

    assert features.shape == (5, 64)
    np.testing.assert_allclose(features[:, :32], 30.0)
    np.testing.assert_allclose(features[:, 32:], 40.0)
    assert [cfg["r_lrf"] for cfg in _FakeGeDi.configs] == [30.0, 40.0]
