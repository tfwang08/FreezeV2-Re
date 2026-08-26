from __future__ import annotations
import numpy as np
from .geometry import kabsch


def sparse_grid_pixels(mask: np.ndarray, grid_size: int = 16) -> np.ndarray:
    """Sample at most grid_size^2 foreground pixels on a regular bbox grid."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x_edges = np.linspace(x0, x1, grid_size + 1)
    y_edges = np.linspace(y0, y1, grid_size + 1)
    out = []
    for gy in range(grid_size):
        ya, yb = int(np.floor(y_edges[gy])), int(np.ceil(y_edges[gy + 1]))
        for gx in range(grid_size):
            xa, xb = int(np.floor(x_edges[gx])), int(np.ceil(x_edges[gx + 1]))
            yy, xx = np.nonzero(mask[ya:yb, xa:xb])
            if len(xx) == 0:
                continue
            xx = xx + xa
            yy = yy + ya
            cx = 0.5 * (x_edges[gx] + x_edges[gx + 1] - 1)
            cy = 0.5 * (y_edges[gy] + y_edges[gy + 1] - 1)
            j = np.argmin((xx - cx) ** 2 + (yy - cy) ** 2)
            out.append((int(xx[j]), int(yy[j])))
    return np.asarray(out, dtype=np.int64)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _score_hypothesis(pose, query_points, query_features, target_points, target_features,
                      candidate_query_indices, inlier_threshold: float) -> float:
    """FreeZeV2 Eq. (5) feature-aware coarse-pose score.

    Every top-k correspondence whose transformed query point is inside the
    geometric inlier threshold contributes its cosine similarity. The total is
    normalized by the number of sparse target points, not by the number of
    correspondences. Consequently the score can be larger than 1 when several
    candidates for a target point are geometric inliers.
    """
    cand_pts = query_points[candidate_query_indices]
    transformed = cand_pts @ pose[:3, :3].T + pose[:3, 3]
    dist = np.linalg.norm(transformed - target_points[:, None, :], axis=2)
    qf = query_features[candidate_query_indices]
    tf = target_features[:, None, :]
    cosine = np.sum(qf * tf, axis=2)
    inliers = dist < float(inlier_threshold)
    return float(np.sum(cosine[inliers]) / len(target_points))


def feature_aware_ransac(query_points: np.ndarray, query_features: np.ndarray,
                         target_points: np.ndarray, target_features: np.ndarray,
                         candidate_query_indices: np.ndarray, inlier_threshold: float,
                         iterations: int = 10_000, seed: int = 0) -> tuple[np.ndarray, float]:
    """FreeZeV2-style feature-aware 3D-3D RANSAC."""
    qp = np.asarray(query_points, dtype=np.float64)
    tp = np.asarray(target_points, dtype=np.float64)
    qi = np.asarray(candidate_query_indices, dtype=np.int64)
    qf = _normalize(query_features)
    tf = _normalize(target_features)
    if len(tp) < 3 or qi.shape[0] != len(tp):
        raise ValueError("need at least 3 target points and one candidate row per target")
    if qi.ndim != 2 or qi.shape[1] == 0:
        raise ValueError("candidate_query_indices must have shape NxK with K > 0")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    if float(inlier_threshold) <= 0:
        raise ValueError("inlier_threshold must be positive")
    rng = np.random.default_rng(seed)
    best_pose = np.eye(4, dtype=np.float64)
    best_score = -np.inf
    k = qi.shape[1]
    for _ in range(int(iterations)):
        ti = rng.choice(len(tp), size=3, replace=False)
        ci = rng.integers(0, k, size=3)
        qids = qi[ti, ci]
        src = qp[qids]
        if len(np.unique(qids)) < 3:
            continue
        if np.linalg.matrix_rank(src[1:] - src[:1]) < 2 or np.linalg.matrix_rank(tp[ti][1:] - tp[ti][:1]) < 2:
            continue
        try:
            R, t = kabsch(src, tp[ti])
        except np.linalg.LinAlgError:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = t
        score = _score_hypothesis(pose, qp, qf, tp, tf, qi, inlier_threshold)
        if score > best_score:
            best_score = score
            best_pose = pose
    return best_pose, float(best_score)
