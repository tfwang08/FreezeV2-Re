from __future__ import annotations
import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def topk_cosine_matches(target_features: np.ndarray, query_features: np.ndarray, k: int = 10):
    target = _l2_normalize(target_features)
    query = _l2_normalize(query_features)
    k = min(int(k), len(query))
    sim = target @ query.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    vals = np.take_along_axis(sim, idx, axis=1)
    return idx, vals
