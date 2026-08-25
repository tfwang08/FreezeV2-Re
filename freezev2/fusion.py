from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualPCA:
    mean: np.ndarray
    components: np.ndarray


def _as_feature_matrix(features, name: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"{name} must have shape NxD")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} must contain only finite values")
    return features


def fit_visual_pca(query_visual, output_dim: int = 64) -> VisualPCA:
    """Fit PCA on query visual descriptors only, as specified by FreeZeV2."""
    query_visual = _as_feature_matrix(query_visual, "query_visual")
    output_dim = int(output_dim)
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if output_dim > min(query_visual.shape):
        raise ValueError("output_dim exceeds the PCA rank supported by query_visual")

    mean = query_visual.mean(axis=0)
    centered = query_visual - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:output_dim].copy()
    return VisualPCA(mean=mean, components=components)


def apply_visual_pca(features, state: VisualPCA) -> np.ndarray:
    features = _as_feature_matrix(features, "features")
    mean = np.asarray(state.mean, dtype=np.float64)
    components = np.asarray(state.components, dtype=np.float64)
    if mean.ndim != 1 or components.ndim != 2:
        raise ValueError("invalid PCA state")
    if features.shape[1] != mean.shape[0] or components.shape[1] != mean.shape[0]:
        raise ValueError("feature dimension does not match PCA state")
    return ((features - mean) @ components.T).astype(np.float32)


def l2_normalize_rows(features, eps: float = 1e-12) -> np.ndarray:
    features = _as_feature_matrix(features, "features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe = np.maximum(norms, float(eps))
    return (features / safe).astype(np.float32)


def fuse_visual_geometric(
    visual_features,
    geometric_features,
    pca_state: VisualPCA,
) -> np.ndarray:
    """Apply Eq. (1): [norm(PCA(f_vis)), norm(f_geo)]."""
    geometric_features = _as_feature_matrix(geometric_features, "geometric_features")
    visual_pca = apply_visual_pca(visual_features, pca_state)
    if visual_pca.shape[0] != geometric_features.shape[0]:
        raise ValueError("visual and geometric features must contain the same number of points")
    if visual_pca.shape[1] != geometric_features.shape[1]:
        raise ValueError("PCA visual and geometric branches must have the same dimension")

    visual_norm = l2_normalize_rows(visual_pca)
    geometric_norm = l2_normalize_rows(geometric_features)
    return np.concatenate([visual_norm, geometric_norm], axis=1).astype(np.float32)
