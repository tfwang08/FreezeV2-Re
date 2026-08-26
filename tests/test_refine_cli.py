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


def test_refine_pose_saves_icp_and_final_scores(tmp_path, monkeypatch, capsys):
    query_points = np.array([
        [-20.0, -10.0, 0.0],
        [15.0, -12.0, 4.0],
        [-8.0, 20.0, 7.0],
        [18.0, 16.0, -5.0],
        [0.0, 0.0, 15.0],
        [-15.0, 8.0, 22.0],
        [12.0, -18.0, 20.0],
        [22.0, 5.0, 12.0],
    ], dtype=np.float32)
    query_features = np.eye(8, dtype=np.float32)

    angle = np.deg2rad(5.0)
    gt_R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    gt_t = np.array([100.0, -50.0, 600.0], dtype=np.float64)
    dense_target = query_points.astype(np.float64) @ gt_R.T + gt_t
    sparse_ids = np.array([0, 2, 4, 6], dtype=np.int64)
    target_points = dense_target[sparse_ids].astype(np.float32)
    target_features = query_features[sparse_ids].copy()

    coarse_pose = np.eye(4, dtype=np.float64)
    coarse_pose[:3, :3] = gt_R
    coarse_pose[:3, 3] = gt_t + np.array([1.0, -1.0, 0.5])
    candidate_idx = sparse_ids[:, None]

    query_cache = tmp_path / "query.npz"
    target_cache = tmp_path / "target.npz"
    coarse_cache = tmp_path / "coarse.npz"
    output = tmp_path / "fine.npz"
    np.savez_compressed(
        query_cache,
        query_points=query_points,
        fused_features=query_features,
        diameter=np.float32(100.0),
    )
    np.savez_compressed(
        target_cache,
        sparse_points=target_points,
        dense_points=dense_target.astype(np.float32),
        target_features=target_features,
        scene_id=np.int32(2),
        im_id=np.int32(3),
        obj_id=np.int32(1),
    )
    np.savez_compressed(
        coarse_cache,
        coarse_pose=coarse_pose,
        coarse_score=np.float64(0.8),
        candidate_query_indices=candidate_idx,
        scene_id=np.int32(2),
        im_id=np.int32(3),
        obj_id=np.int32(1),
    )

    bop_root = tmp_path / "data" / "bop"
    scene_dir = bop_root / "lmo" / "test" / "000002"
    scene_dir.mkdir(parents=True)
    (scene_dir / "scene_gt.json").write_text(json.dumps({
        "3": [{
            "obj_id": 1,
            "cam_R_m2c": gt_R.reshape(-1).tolist(),
            "cam_t_m2c": gt_t.tolist(),
        }]
    }))

    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "refine-pose",
        "--dataset", "lmo",
        "--scene-id", "2",
        "--im-id", "3",
        "--obj-id", "1",
        "--query-cache", str(query_cache),
        "--target-cache", str(target_cache),
        "--coarse-cache", str(coarse_cache),
        "--bop-root", str(bop_root),
        "--gt-id", "0",
        "--output", str(output),
    ])

    run_bop.main()

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["fine_pose"][:3, :3], gt_R, atol=1e-7)
        np.testing.assert_allclose(data["fine_pose"][:3, 3], gt_t, atol=1e-7)
        np.testing.assert_allclose(float(data["icp_threshold"]), 3.0)
        np.testing.assert_allclose(float(data["coarse_feature_score"]), 0.8)
        np.testing.assert_allclose(float(data["fine_feature_score"]), 1.0, atol=1e-7)
        np.testing.assert_allclose(float(data["icp_score"]), 1.0, atol=1e-7)
        np.testing.assert_allclose(float(data["final_score"]), 0.8, atol=1e-7)
        assert int(data["icp_max_iterations"]) == 30
        np.testing.assert_allclose([data["alpha"], data["beta"], data["gamma"]], [1.0, 1.0, 1.0])
        # The real target cache is float32, so an otherwise exact synthetic
        # rigid transform retains only float32 point precision.  GT metrics in
        # the low-microdegree/sub-micron range are already numerical zero for
        # this interface and should not be tested against float64 exactness.
        np.testing.assert_allclose(float(data["fine_rotation_error_deg"]), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(data["fine_translation_error_mm"]), 0.0, atol=1e-5)

    report = json.loads(capsys.readouterr().out)
    np.testing.assert_allclose(report["icp_threshold"], 3.0)
    np.testing.assert_allclose(report["coarse_feature_score"], 0.8)
    np.testing.assert_allclose(report["fine_feature_score"], 1.0, atol=1e-7)
    np.testing.assert_allclose(report["icp_score"], 1.0, atol=1e-7)
    np.testing.assert_allclose(report["final_score"], 0.8, atol=1e-7)
    np.testing.assert_allclose(report["fine_rotation_error_deg"], 0.0, atol=1e-5)
    np.testing.assert_allclose(report["fine_translation_error_mm"], 0.0, atol=1e-5)
