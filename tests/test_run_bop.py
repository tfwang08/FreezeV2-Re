import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


_RUN_BOP_PATH = Path(__file__).resolve().parents[1] / "run_bop.py"
_RUN_BOP_SPEC = importlib.util.spec_from_file_location("run_bop", _RUN_BOP_PATH)
assert _RUN_BOP_SPEC is not None and _RUN_BOP_SPEC.loader is not None
run_bop = importlib.util.module_from_spec(_RUN_BOP_SPEC)
_RUN_BOP_SPEC.loader.exec_module(run_bop)


def test_extract_query_gedi_reads_model_diameter_and_saves_64d(
    tmp_path,
    monkeypatch,
):
    bop_root = tmp_path / "data" / "bop"
    models = bop_root / "lmo" / "models"
    models.mkdir(parents=True)
    (models / "models_info.json").write_text(
        json.dumps({"1": {"diameter": 100.0}})
    )

    cache = tmp_path / "onboarding.npz"
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    np.savez_compressed(cache, query_points=points)
    output = tmp_path / "gedi.npz"
    calls = {}

    class FakeExtractor:
        def __init__(self, checkpoint, gedi_root, seed=0):
            calls["init"] = (Path(checkpoint), Path(gedi_root), seed)

        def encode(self, pts, cloud, object_diameter):
            calls["diameter"] = object_diameter
            np.testing.assert_array_equal(pts, points)
            np.testing.assert_array_equal(cloud, points)
            return np.concatenate([
                np.full((len(pts), 32), 30.0, dtype=np.float32),
                np.full((len(pts), 32), 40.0, dtype=np.float32),
            ], axis=1)

    monkeypatch.setattr(run_bop, "GediExtractor", FakeExtractor)
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "extract-query-gedi",
        "--dataset",
        "lmo",
        "--obj-id",
        "1",
        "--bop-root",
        str(bop_root),
        "--onboarding-cache",
        str(cache),
        "--gedi-root",
        str(tmp_path / "gedi"),
        "--checkpoint",
        str(tmp_path / "checkpoint.tar"),
        "--output",
        str(output),
        "--seed",
        "7",
    ])

    run_bop.main()

    assert calls["diameter"] == 100.0
    assert calls["init"][2] == 7
    with np.load(output, allow_pickle=False) as data:
        assert data["query_points"].shape == (5, 3)
        assert data["geometric_features_30"].shape == (5, 32)
        assert data["geometric_features_40"].shape == (5, 32)
        assert data["geometric_features"].shape == (5, 64)
        np.testing.assert_allclose(data["radii"], [30.0, 40.0])


def test_fuse_query_features_saves_128d_and_query_pca(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(80, 3)).astype(np.float32)
    visual = rng.normal(size=(80, 96)).astype(np.float32)
    geometric = rng.normal(size=(80, 64)).astype(np.float32)

    visual_path = tmp_path / "visual.npz"
    geometric_path = tmp_path / "gedi.npz"
    output = tmp_path / "query.npz"
    np.savez_compressed(
        visual_path,
        query_points=points,
        visual_features=visual,
    )
    np.savez_compressed(
        geometric_path,
        query_points=points,
        geometric_features=geometric,
        diameter=np.float32(100.0),
    )

    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "fuse-query-features",
        "--dataset",
        "lmo",
        "--obj-id",
        "1",
        "--visual",
        str(visual_path),
        "--geometric",
        str(geometric_path),
        "--output",
        str(output),
    ])

    run_bop.main()

    with np.load(output, allow_pickle=False) as data:
        fused = np.asarray(data["fused_features"], dtype=np.float32)
        assert data["query_points"].shape == (80, 3)
        assert fused.shape == (80, 128)
        assert data["pca_mean"].shape == (96,)
        assert data["pca_components"].shape == (64, 96)
        assert int(data["pca_dim"]) == 64
        assert float(data["diameter"]) == 100.0
        np.testing.assert_allclose(
            np.linalg.norm(fused[:, :64], axis=1),
            1.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(fused[:, 64:], axis=1),
            1.0,
            atol=1e-5,
        )
        assert np.isfinite(fused).all()
