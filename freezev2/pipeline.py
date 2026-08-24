from __future__ import annotations
import numpy as np
from .matching import topk_cosine_matches
from .pose import feature_aware_ransac


def estimate_pose_from_features(query_points: np.ndarray, query_features: np.ndarray,
                                target_points: np.ndarray, target_features: np.ndarray,
                                object_diameter: float, k: int = 10,
                                iterations: int = 10_000, seed: int = 0):
    candidate_idx, _ = topk_cosine_matches(target_features, query_features, k=k)
    return feature_aware_ransac(
        query_points=query_points,
        query_features=query_features,
        target_points=target_points,
        target_features=target_features,
        candidate_query_indices=candidate_idx,
        inlier_threshold=0.03 * float(object_diameter),
        iterations=iterations,
        seed=seed,
    )
