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


def _triplet_edges_compatible(
    source: np.ndarray,
    target: np.ndarray,
    similarity_threshold: float,
) -> bool:
    """Check rigid-triplet consistency using relative pairwise edge lengths.

    ``similarity_threshold`` is dimensionless and follows the conventional
    Open3D-style edge-length checker: for every corresponding edge, each edge
    length must exceed ``threshold`` times the other.  A value near one is
    stricter.  FreeZeV2 describes relative edge-length pruning but does not
    publish this numeric threshold, so callers must keep it explicit.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    threshold = float(similarity_threshold)
    if source.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("source and target triplets must each have shape 3x3")
    if not np.isfinite(threshold) or not (0.0 < threshold <= 1.0):
        raise ValueError("similarity_threshold must be finite and in (0, 1]")

    pairs = ((0, 1), (0, 2), (1, 2))
    source_edges = np.asarray([
        np.linalg.norm(source[a] - source[b]) for a, b in pairs
    ])
    target_edges = np.asarray([
        np.linalg.norm(target[a] - target[b]) for a, b in pairs
    ])
    if (
        not np.isfinite(source_edges).all()
        or not np.isfinite(target_edges).all()
        or np.any(source_edges <= 0.0)
        or np.any(target_edges <= 0.0)
    ):
        return False
    return bool(np.all(
        (source_edges > threshold * target_edges)
        & (target_edges > threshold * source_edges)
    ))


def _inlier_mask(
    pose: np.ndarray,
    query_points: np.ndarray,
    target_points: np.ndarray,
    candidate_query_indices: np.ndarray,
    inlier_threshold: float,
) -> np.ndarray:
    cand_pts = query_points[candidate_query_indices]
    transformed = cand_pts @ pose[:3, :3].T + pose[:3, 3]
    dist = np.linalg.norm(transformed - target_points[:, None, :], axis=2)
    return dist < float(inlier_threshold)


def _score_hypothesis(pose, query_points, query_features, target_points, target_features,
                      candidate_query_indices, inlier_threshold: float) -> float:
    """FreeZeV2 Eq. (5) feature-aware coarse-pose score.

    Every top-k correspondence whose transformed query point is inside the
    geometric inlier threshold contributes its cosine similarity. The total is
    normalized by the number of sparse target points, not by the number of
    correspondences. Consequently the score can be larger than 1 when several
    candidates for a target point are geometric inliers.
    """
    inliers = _inlier_mask(
        pose,
        query_points,
        target_points,
        candidate_query_indices,
        inlier_threshold,
    )
    qf = query_features[candidate_query_indices]
    tf = target_features[:, None, :]
    cosine = np.sum(qf * tf, axis=2)
    return float(np.sum(cosine[inliers]) / len(target_points))


def feature_aware_ransac(query_points: np.ndarray, query_features: np.ndarray,
                         target_points: np.ndarray, target_features: np.ndarray,
                         candidate_query_indices: np.ndarray, inlier_threshold: float,
                         iterations: int = 10_000, seed: int = 0,
                         edge_similarity_threshold: float = 0.9,
                         return_debug: bool = False):
    """FreeZeV2-style feature-aware 3D-3D RANSAC.

    The published geometric inlier threshold stays independent of triplet
    pruning. ``edge_similarity_threshold`` is an explicit reverse-engineering
    parameter for the paper's unpublished relative-edge consistency check;
    0.9 is an Open3D-style candidate rather than a paper-published constant.
    ``return_debug=True`` appends the winning sampled correspondences and
    pruning/inlier statistics.
    """
    qp = np.asarray(query_points, dtype=np.float64)
    tp = np.asarray(target_points, dtype=np.float64)
    qi = np.asarray(candidate_query_indices, dtype=np.int64)
    qf = _normalize(query_features)
    tf = _normalize(target_features)
    if len(tp) < 3 or qi.shape[0] != len(tp):
        raise ValueError("need at least 3 target points and one candidate row per target")
    if qi.ndim != 2 or qi.shape[1] == 0:
        raise ValueError("candidate_query_indices must have shape NxK with K > 0")
    if np.any(qi < 0) or np.any(qi >= len(qp)):
        raise ValueError("candidate_query_indices contains an out-of-range query index")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    if float(inlier_threshold) <= 0:
        raise ValueError("inlier_threshold must be positive")
    edge_similarity_threshold = float(edge_similarity_threshold)
    if (
        not np.isfinite(edge_similarity_threshold)
        or not (0.0 < edge_similarity_threshold <= 1.0)
    ):
        raise ValueError("edge_similarity_threshold must be finite and in (0, 1]")

    rng = np.random.default_rng(seed)
    best_pose = np.eye(4, dtype=np.float64)
    best_score = -np.inf
    best_target_indices = np.empty(0, dtype=np.int64)
    best_candidate_columns = np.empty(0, dtype=np.int64)
    best_query_indices = np.empty(0, dtype=np.int64)
    k = qi.shape[1]
    degenerate_triplets = 0
    edge_pruned_triplets = 0
    valid_hypotheses = 0

    for _ in range(int(iterations)):
        ti = rng.choice(len(tp), size=3, replace=False)
        ci = rng.integers(0, k, size=3)
        qids = qi[ti, ci]
        src = qp[qids]
        dst = tp[ti]
        if len(np.unique(qids)) < 3:
            degenerate_triplets += 1
            continue
        if (
            np.linalg.matrix_rank(src[1:] - src[:1]) < 2
            or np.linalg.matrix_rank(dst[1:] - dst[:1]) < 2
        ):
            degenerate_triplets += 1
            continue
        if not _triplet_edges_compatible(
            src,
            dst,
            similarity_threshold=edge_similarity_threshold,
        ):
            edge_pruned_triplets += 1
            continue
        try:
            R, t = kabsch(src, dst)
        except np.linalg.LinAlgError:
            degenerate_triplets += 1
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = t
        valid_hypotheses += 1
        score = _score_hypothesis(pose, qp, qf, tp, tf, qi, inlier_threshold)
        if score > best_score:
            best_score = score
            best_pose = pose
            best_target_indices = np.asarray(ti, dtype=np.int64).copy()
            best_candidate_columns = np.asarray(ci, dtype=np.int64).copy()
            best_query_indices = np.asarray(qids, dtype=np.int64).copy()

    if np.isfinite(best_score):
        winning_inliers = _inlier_mask(
            best_pose,
            qp,
            tp,
            qi,
            inlier_threshold,
        )
        inlier_count = int(winning_inliers.sum())
        inlier_target_count = int(np.any(winning_inliers, axis=1).sum())
    else:
        inlier_count = 0
        inlier_target_count = 0

    if not return_debug:
        return best_pose, float(best_score)

    debug = {
        "winning_target_indices": best_target_indices,
        "winning_candidate_columns": best_candidate_columns,
        "winning_query_indices": best_query_indices,
        "inlier_count": inlier_count,
        "inlier_target_count": inlier_target_count,
        "edge_similarity_threshold": edge_similarity_threshold,
        "degenerate_triplets": int(degenerate_triplets),
        "edge_pruned_triplets": int(edge_pruned_triplets),
        "valid_hypotheses": int(valid_hypotheses),
        "iterations": int(iterations),
        "seed": int(seed),
    }
    return best_pose, float(best_score), debug
