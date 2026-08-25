#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from freezev2.fusion import fit_visual_pca, fuse_visual_geometric


def load_npz(path):
    return np.load(Path(path), allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", required=True)
    parser.add_argument("--geometric", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visual-key", default="visual_features")
    parser.add_argument("--geometric-key", default="geometric_features")
    parser.add_argument("--points-key", default="query_points")
    parser.add_argument("--pca-dim", type=int, default=64)
    args = parser.parse_args()

    with load_npz(args.visual) as visual_data:
        points_visual = np.asarray(visual_data[args.points_key], dtype=np.float32)
        visual = np.asarray(visual_data[args.visual_key], dtype=np.float32)

    with load_npz(args.geometric) as geometric_data:
        points_geometric = np.asarray(
            geometric_data[args.points_key], dtype=np.float32
        )
        geometric = np.asarray(
            geometric_data[args.geometric_key], dtype=np.float32
        )

    if points_visual.shape != points_geometric.shape or not np.allclose(
        points_visual, points_geometric, atol=1e-5, rtol=0.0
    ):
        raise ValueError("visual and geometric caches do not describe the same points")
    if len(visual) != len(points_visual) or len(geometric) != len(points_visual):
        raise ValueError("feature caches and query points have inconsistent lengths")

    pca = fit_visual_pca(visual, output_dim=args.pca_dim)
    fused = fuse_visual_geometric(visual, geometric, pca)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        query_points=points_visual,
        fused_features=fused,
        pca_mean=pca.mean.astype(np.float32),
        pca_components=pca.components.astype(np.float32),
        visual_dim=np.int32(visual.shape[1]),
        geometric_dim=np.int32(geometric.shape[1]),
        pca_dim=np.int32(args.pca_dim),
    )

    print("query points:", points_visual.shape)
    print("visual:", visual.shape)
    print("geometric:", geometric.shape)
    print("PCA components:", pca.components.shape)
    print("fused:", fused.shape)
    print(
        "branch norms:",
        float(np.linalg.norm(fused[:, : args.pca_dim], axis=1).min()),
        float(np.linalg.norm(fused[:, : args.pca_dim], axis=1).max()),
        float(np.linalg.norm(fused[:, args.pca_dim :], axis=1).min()),
        float(np.linalg.norm(fused[:, args.pca_dim :], axis=1).max()),
    )
    print("finite:", bool(np.isfinite(fused).all()))
    print("saved:", output)


if __name__ == "__main__":
    main()
