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


def test_estimate_coarse_pose_uses_paper_defaults_and_gt_metrics(
    tmp_path,
    monkeypatch,
    capsys,
):
    rng = np.random.default_rng(5)
    query_points = rng.normal(size=(12, 3)).astype(np.float32)
    query_features = rng.normal(size=(12, 128)).astype(np.float32)
    query_features /= np.linalg.norm(query_features, axis=1, keepdims=True)
    target_points = rng.normal(size=(4, 3)).astype(np.float32)
    target_features = rng.normal(size=(4, 128)).astype(np.float32)
    target_features /= np.linalg.norm(target_features, axis=1, keepdims=True)

    query_cache = tmp_path / "query.npz"
    target_cache = tmp_path / "target.npz"
    output = tmp_path / "coarse_pose.npz"
    np.savez_compressed(
        query_cache,
        query_points=query_points,
        fused_features=query_features,
        diameter=np.float32(100.0),
    )
    np.savez_compressed(
        target_cache,
        sparse_points=target_points,
        target_features=target_features,
        scene_id=np.int32(2),
        im_id=np.int32(3),
        obj_id=np.int32(1),
    )

    bop_root = tmp_path / "data" / "bop"
    scene_dir = bop_root / "lmo" / "test" / "000002"
    scene_dir.mkdir(parents=True)
    gt_R = np.eye(3, dtype=np.float64)
    gt_t = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    (scene_dir / "scene_gt.json").write_text(json.dumps({
        "3": [{
            "obj_id": 1,
            "cam_R_m2c": gt_R.reshape(-1).tolist(),
            "cam_t_m2c": gt_t.tolist(),
        }]
    }))

    candidate_idx = np.tile(np.arange(10, dtype=np.int64), (4, 1))
    candidate_sim = np.tile(
        np.linspace(0.95, 0.50, 10, dtype=np.float64),
        (4, 1),
    )
    calls = {}

    def fake_topk(target, query, k=10):
        calls["topk"] = (target.shape, query.shape, k)
        return candidate_idx.copy(), candidate_sim.copy()

    def fake_estimate(
        query_points_arg,
        query_features_arg,
        target_points_arg,
        target_features_arg,
        object_diameter,
        k=10,
        iterations=10_000,
        seed=0,
        edge_tolerance=None,
        return_debug=False,
    ):
        np.testing.assert_array_equal(query_points_arg, query_points)
        np.testing.assert_array_equal(query_features_arg, query_features)
        np.testing.assert_array_equal(target_points_arg, target_points)
        np.testing.assert_array_equal(target_features_arg, target_features)
        calls["estimate"] = (
            object_diameter,
            k,
            iterations,
            seed,
            edge_tolerance,
            return_debug,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = gt_R
        pose[:3, 3] = gt_t
        debug = {
            "winning_target_indices": np.array([0, 1, 2], dtype=np.int64),
            "winning_candidate_columns": np.array([0, 1, 2], dtype=np.int64),
            "winning_query_indices": np.array([0, 1, 2], dtype=np.int64),
            "inlier_count": 7,
            "inlier_target_count": 4,
            "edge_tolerance": 3.0,
            "degenerate_triplets": 11,
            "edge_pruned_triplets": 23,
            "valid_hypotheses": 9966,
        }
        return pose, 2.75, debug

    monkeypatch.setattr(run_bop, "topk_cosine_matches", fake_topk, raising=False)
    monkeypatch.setattr(run_bop, "estimate_pose_from_features", fake_estimate, raising=False)
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "estimate-coarse-pose",
        "--dataset",
        "lmo",
        "--scene-id",
        "2",
        "--im-id",
        "3",
        "--obj-id",
        "1",
        "--query-cache",
        str(query_cache),
        "--target-cache",
        str(target_cache),
        "--bop-root",
        str(bop_root),
        "--gt-id",
        "0",
        "--output",
        str(output),
    ])

    run_bop.main()

    assert calls["topk"] == ((4, 128), (12, 128), 10)
    assert calls["estimate"] == (100.0, 10, 10_000, 0, 3.0, True)

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["coarse_pose"], np.block([
            [gt_R, gt_t[:, None]],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]))
        np.testing.assert_array_equal(data["candidate_query_indices"], candidate_idx)
        np.testing.assert_allclose(data["candidate_similarities"], candidate_sim)
        assert float(data["coarse_score"]) == 2.75
        np.testing.assert_allclose(float(data["inlier_threshold"]), 3.0)
        np.testing.assert_allclose(float(data["edge_tolerance"]), 3.0)
        assert int(data["top_k"]) == 10
        assert int(data["iterations"]) == 10_000
        assert int(data["gt_id"]) == 0
        assert int(data["inlier_count"]) == 7
        assert int(data["inlier_target_count"]) == 4
        assert int(data["degenerate_triplets"]) == 11
        assert int(data["edge_pruned_triplets"]) == 23
        assert int(data["valid_hypotheses"]) == 9966
        np.testing.assert_array_equal(data["winning_target_indices"], [0, 1, 2])
        np.testing.assert_array_equal(data["winning_candidate_columns"], [0, 1, 2])
        np.testing.assert_array_equal(data["winning_query_indices"], [0, 1, 2])
        np.testing.assert_allclose(float(data["rotation_error_deg"]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(data["translation_error_mm"]), 0.0, atol=1e-10)

    report = json.loads(capsys.readouterr().out)
    assert report["top_k"] == 10
    assert report["iterations"] == 10_000
    assert report["inlier_count"] == 7
    assert report["inlier_target_count"] == 4
    assert report["degenerate_triplets"] == 11
    assert report["edge_pruned_triplets"] == 23
    assert report["valid_hypotheses"] == 9966
    np.testing.assert_allclose(report["inlier_threshold"], 3.0)
    np.testing.assert_allclose(report["edge_tolerance"], 3.0)
    np.testing.assert_allclose(report["rotation_error_deg"], 0.0, atol=1e-10)
    np.testing.assert_allclose(report["translation_error_mm"], 0.0, atol=1e-10)
