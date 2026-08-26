from __future__ import annotations
import numpy as np
from .matching import topk_cosine_matches
from .pose import feature_aware_ransac


def estimate_pose_from_features(query_points: np.ndarray, query_features: np.ndarray,
                                target_points: np.ndarray, target_features: np.ndarray,
                                object_diameter: float, k: int = 10,
                                iterations: int = 10_000, seed: int = 0,
                                edge_similarity_threshold: float = 0.9,
                                return_debug: bool = False):
    candidate_idx, _ = topk_cosine_matches(target_features, query_features, k=k)
    inlier_threshold = 0.03 * float(object_diameter)
    return feature_aware_ransac(
        query_points=query_points,
        query_features=query_features,
        target_points=target_points,
        target_features=target_features,
        candidate_query_indices=candidate_idx,
        inlier_threshold=inlier_threshold,
        iterations=iterations,
        seed=seed,
        edge_similarity_threshold=edge_similarity_threshold,
        return_debug=return_debug,
    )
