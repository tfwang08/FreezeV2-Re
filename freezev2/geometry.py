from __future__ import annotations
import numpy as np


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("src and dst must be Nx3 arrays with N>=3")
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c
    return R, t


def backproject_depth(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None):
    depth = np.asarray(depth)
    K = np.asarray(K, dtype=np.float64)
    if mask is None:
        mask = depth > 0
    else:
        mask = np.asarray(mask, dtype=bool) & (depth > 0)
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    uv = np.stack([u, v], axis=1)
    return pts, uv
