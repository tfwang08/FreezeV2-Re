from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pose import feature_pose_score, point_to_point_icp


def _rotation_error_deg(predicted: np.ndarray, reference: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(3, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(3, 3)
    delta = predicted @ reference.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _check_cache_ids(data, *, scene_id: int, im_id: int, obj_id: int, label: str) -> None:
    for key, expected in (
        ("scene_id", scene_id),
        ("im_id", im_id),
        ("obj_id", obj_id),
    ):
        if key in data and int(data[key]) != int(expected):
            raise ValueError(
                f"{label} {key}={int(data[key])} does not match CLI {expected}"
            )


def refine_pose_cache(
    *,
    dataset: str,
    scene_id: int,
    im_id: int,
    obj_id: int,
    target_cache: Path,
    query_cache: Path | None = None,
    coarse_cache: Path | None = None,
    bop_root: Path = Path("data/bop"),
    split: str = "test",
    icp_max_iterations: int = 30,
    gt_id: int | None = None,
    output: Path | None = None,
) -> dict:
    """Refine one saved coarse pose and persist FreeZeV2 fine-stage scores.

    The published fine stage uses ``tau_ICP = 0.03 * diameter`` for dense
    point-cloud ICP and combines coarse feature score, fine feature score and
    ICP inlier ratio with exponents alpha=beta=gamma=1.  The paper does not
    publish the ICP optimizer's convergence settings, so ``icp_max_iterations``
    stays explicit and is persisted in the output cache.
    """
    scene_id = int(scene_id)
    im_id = int(im_id)
    obj_id = int(obj_id)
    icp_max_iterations = int(icp_max_iterations)
    if scene_id < 0 or im_id < 0:
        raise ValueError("scene_id and im_id must be non-negative")
    if obj_id <= 0:
        raise ValueError("obj_id must be positive")
    if icp_max_iterations <= 0:
        raise ValueError("icp_max_iterations must be positive")
    if gt_id is not None and int(gt_id) < 0:
        raise ValueError("gt_id must be non-negative")

    query_cache = query_cache or (
        Path("outputs/features") / f"{dataset}_obj_{obj_id:06d}_query.npz"
    )
    coarse_cache = coarse_cache or (
        Path("outputs/poses")
        / f"{dataset}_scene_{scene_id:06d}_im_{im_id:06d}_obj_{obj_id:06d}_coarse.npz"
    )
    target_cache = Path(target_cache)
    query_cache = Path(query_cache)
    coarse_cache = Path(coarse_cache)
    for path, label in (
        (query_cache, "query cache"),
        (target_cache, "target cache"),
        (coarse_cache, "coarse cache"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    with np.load(query_cache, allow_pickle=False) as data:
        required = ("query_points", "fused_features", "diameter")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError("query cache is missing: " + ", ".join(missing))
        query_points = np.asarray(data["query_points"], dtype=np.float64)
        query_features = np.asarray(data["fused_features"], dtype=np.float64)
        diameter = float(data["diameter"])

    with np.load(target_cache, allow_pickle=False) as data:
        required = ("sparse_points", "dense_points", "target_features")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError("target cache is missing: " + ", ".join(missing))
        _check_cache_ids(
            data,
            scene_id=scene_id,
            im_id=im_id,
            obj_id=obj_id,
            label="target cache",
        )
        sparse_points = np.asarray(data["sparse_points"], dtype=np.float64)
        dense_points = np.asarray(data["dense_points"], dtype=np.float64)
        target_features = np.asarray(data["target_features"], dtype=np.float64)

    with np.load(coarse_cache, allow_pickle=False) as data:
        required = ("coarse_pose", "coarse_score", "candidate_query_indices")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError("coarse cache is missing: " + ", ".join(missing))
        _check_cache_ids(
            data,
            scene_id=scene_id,
            im_id=im_id,
            obj_id=obj_id,
            label="coarse cache",
        )
        coarse_pose = np.asarray(data["coarse_pose"], dtype=np.float64)
        coarse_feature_score = float(data["coarse_score"])
        candidate_query_indices = np.asarray(
            data["candidate_query_indices"], dtype=np.int64
        )

    if query_points.ndim != 2 or query_points.shape[1] != 3 or len(query_points) < 3:
        raise ValueError("query_points must have shape Nx3 with N >= 3")
    if sparse_points.ndim != 2 or sparse_points.shape[1] != 3 or len(sparse_points) == 0:
        raise ValueError("target sparse_points must have non-empty shape Nx3")
    if dense_points.ndim != 2 or dense_points.shape[1] != 3 or len(dense_points) < 3:
        raise ValueError("target dense_points must have shape Mx3 with M >= 3")
    if query_features.ndim != 2 or len(query_features) != len(query_points):
        raise ValueError("query fused_features must have shape NxD matching query_points")
    if target_features.ndim != 2 or len(target_features) != len(sparse_points):
        raise ValueError("target_features must have shape NxD matching sparse_points")
    if query_features.shape[1] != target_features.shape[1]:
        raise ValueError("query and target descriptor dimensions do not match")
    if (
        candidate_query_indices.ndim != 2
        or candidate_query_indices.shape[0] != len(sparse_points)
        or candidate_query_indices.shape[1] == 0
    ):
        raise ValueError(
            "candidate_query_indices must have shape sparse_target_count x K"
        )
    if np.any(candidate_query_indices < 0) or np.any(
        candidate_query_indices >= len(query_points)
    ):
        raise ValueError("candidate_query_indices contains an out-of-range query index")
    if coarse_pose.shape != (4, 4) or not np.isfinite(coarse_pose).all():
        raise ValueError("coarse_pose must be a finite 4x4 matrix")
    if not np.isfinite(coarse_feature_score):
        raise ValueError("coarse_score must be finite")
    if diameter <= 0.0 or not np.isfinite(diameter):
        raise ValueError("query object diameter must be positive and finite")
    for name, array in (
        ("query_points", query_points),
        ("query_features", query_features),
        ("sparse_points", sparse_points),
        ("dense_points", dense_points),
        ("target_features", target_features),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

    icp_threshold = 0.03 * diameter
    fine_pose, icp_score, icp_debug = point_to_point_icp(
        query_points,
        dense_points,
        coarse_pose,
        max_correspondence_distance=icp_threshold,
        max_iterations=icp_max_iterations,
    )
    fine_feature_score = feature_pose_score(
        fine_pose,
        query_points,
        query_features,
        sparse_points,
        target_features,
        candidate_query_indices,
        icp_threshold,
    )

    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    final_score = (
        coarse_feature_score ** alpha
        * fine_feature_score ** beta
        * icp_score ** gamma
    )
    if not np.isfinite(fine_feature_score) or not np.isfinite(final_score):
        raise RuntimeError("fine feature/final score is non-finite")

    fine_R = fine_pose[:3, :3]
    fine_t = fine_pose[:3, 3]
    report = {
        "dataset": dataset,
        "scene_id": scene_id,
        "im_id": im_id,
        "obj_id": obj_id,
        "query_points": list(query_points.shape),
        "sparse_target_points": list(sparse_points.shape),
        "dense_target_points": list(dense_points.shape),
        "descriptor_dim": int(query_features.shape[1]),
        "icp_threshold": float(icp_threshold),
        "icp_max_iterations": icp_max_iterations,
        "icp_iterations_run": int(icp_debug["iterations_run"]),
        "icp_inlier_count": int(icp_debug["inlier_count"]),
        "icp_rmse": float(icp_debug["rmse"]),
        "icp_converged": bool(icp_debug["converged"]),
        "coarse_feature_score": float(coarse_feature_score),
        "fine_feature_score": float(fine_feature_score),
        "icp_score": float(icp_score),
        "final_score": float(final_score),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "R": fine_R.tolist(),
        "t_mm": fine_t.tolist(),
    }

    payload = {
        "fine_pose": fine_pose,
        "coarse_pose": coarse_pose,
        "coarse_feature_score": np.float64(coarse_feature_score),
        "fine_feature_score": np.float64(fine_feature_score),
        "icp_score": np.float64(icp_score),
        "final_score": np.float64(final_score),
        "diameter": np.float32(diameter),
        "icp_threshold": np.float32(icp_threshold),
        "icp_max_iterations": np.int32(icp_max_iterations),
        "icp_iterations_run": np.int32(icp_debug["iterations_run"]),
        "icp_inlier_count": np.int32(icp_debug["inlier_count"]),
        "icp_rmse": np.float64(icp_debug["rmse"]),
        "icp_converged": np.bool_(icp_debug["converged"]),
        "alpha": np.float32(alpha),
        "beta": np.float32(beta),
        "gamma": np.float32(gamma),
        "candidate_query_indices": candidate_query_indices,
        "scene_id": np.int32(scene_id),
        "im_id": np.int32(im_id),
        "obj_id": np.int32(obj_id),
        "query_source": np.array(str(query_cache)),
        "target_source": np.array(str(target_cache)),
        "coarse_source": np.array(str(coarse_cache)),
    }

    if gt_id is not None:
        gt_id = int(gt_id)
        gt_path = (
            Path(bop_root)
            / dataset
            / split
            / f"{scene_id:06d}"
            / "scene_gt.json"
        )
        if not gt_path.is_file():
            raise FileNotFoundError(f"scene GT not found: {gt_path}")
        scene_gt = json.loads(gt_path.read_text())
        annotations = scene_gt.get(str(im_id))
        if annotations is None:
            annotations = scene_gt.get(f"{im_id:06d}")
        if annotations is None:
            raise KeyError(f"image {im_id} missing from {gt_path}")
        if gt_id >= len(annotations):
            raise IndexError(f"gt_id {gt_id} out of range for image {im_id}")
        annotation = annotations[gt_id]
        if int(annotation["obj_id"]) != obj_id:
            raise ValueError(
                f"GT entry obj_id={annotation['obj_id']} does not match obj_id {obj_id}"
            )
        gt_R = np.asarray(annotation["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        gt_t = np.asarray(annotation["cam_t_m2c"], dtype=np.float64).reshape(3)
        coarse_rotation_error_deg = _rotation_error_deg(coarse_pose[:3, :3], gt_R)
        coarse_translation_error_mm = float(
            np.linalg.norm(coarse_pose[:3, 3] - gt_t)
        )
        fine_rotation_error_deg = _rotation_error_deg(fine_R, gt_R)
        fine_translation_error_mm = float(np.linalg.norm(fine_t - gt_t))
        payload.update({
            "gt_id": np.int32(gt_id),
            "gt_R": gt_R,
            "gt_t": gt_t,
            "coarse_rotation_error_deg": np.float64(coarse_rotation_error_deg),
            "coarse_translation_error_mm": np.float64(coarse_translation_error_mm),
            "fine_rotation_error_deg": np.float64(fine_rotation_error_deg),
            "fine_translation_error_mm": np.float64(fine_translation_error_mm),
            "scene_gt_source": np.array(str(gt_path)),
        })
        report.update({
            "gt_id": gt_id,
            "coarse_rotation_error_deg": float(coarse_rotation_error_deg),
            "coarse_translation_error_mm": float(coarse_translation_error_mm),
            "fine_rotation_error_deg": float(fine_rotation_error_deg),
            "fine_translation_error_mm": float(fine_translation_error_mm),
        })

    output = output or (
        Path("outputs/poses")
        / f"{dataset}_scene_{scene_id:06d}_im_{im_id:06d}_obj_{obj_id:06d}_fine.npz"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report["output"] = str(output)
    return report
