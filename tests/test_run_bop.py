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
