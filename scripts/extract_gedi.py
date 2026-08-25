#!/usr/bin/env python3
"""Extract the two FreeZe GeDi scales using the pinned official GeDi repository.

This script is intentionally standalone so it can run in the legacy GeDi Python
environment without installing the main FreezeV2-Re package.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


GEDI_COMMIT = "b3dd86776750d8221f89d39975118da9839b39f7"
SCALES = (0.30, 0.40)


def load_points(path, key):
    path = Path(path)
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            points = data[key]
        finally:
            data.close()
    else:
        points = data
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point input must have shape Nx3")
    if not np.isfinite(points).all():
        raise ValueError("point input must be finite")
    return points


def check_gedi_revision(root):
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != GEDI_COMMIT:
        raise RuntimeError(
            "external GeDi revision mismatch: expected %s, got %s"
            % (GEDI_COMMIT, head)
        )


def gedi_config(radius, checkpoint):
    return {
        "dim": 32,
        "samples_per_batch": 500,
        "samples_per_patch_lrf": 4000,
        "samples_per_patch_out": 512,
        "r_lrf": float(radius),
        "fchkpt_gedi_net": str(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gedi-root", default="external/gedi")
    parser.add_argument(
        "--checkpoint",
        default="external/gedi/data/chkpts/3dmatch/chkpt.tar",
    )
    parser.add_argument("--points", required=True)
    parser.add_argument("--points-key", default="query_points")
    parser.add_argument("--cloud")
    parser.add_argument("--cloud-key", default="query_points")
    parser.add_argument("--diameter", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.diameter <= 0:
        raise ValueError("--diameter must be positive")

    gedi_root = Path(args.gedi_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    check_gedi_revision(gedi_root)

    sys.path.insert(0, str(gedi_root))
    import torch
    from gedi import GeDi

    points = load_points(args.points, args.points_key)
    cloud = (
        points.copy()
        if args.cloud is None
        else load_points(args.cloud, args.cloud_key)
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        raise RuntimeError("official GeDi inference requires CUDA")

    pts_t = torch.from_numpy(points).float()
    cloud_t = torch.from_numpy(cloud).float()

    per_scale = []
    radii = []
    for scale in SCALES:
        radius = scale * args.diameter
        radii.append(radius)
        print("GeDi scale %.2f, radius %.6f" % (scale, radius), flush=True)
        model = GeDi(config=gedi_config(radius, checkpoint))
        descriptors = np.asarray(
            model.compute(pts=pts_t, pcd=cloud_t), dtype=np.float32
        )
        if descriptors.shape != (len(points), 32):
            raise RuntimeError(
                "unexpected GeDi output shape: %s" % (descriptors.shape,)
            )
        if not np.isfinite(descriptors).all():
            raise RuntimeError("GeDi output contains non-finite values")
        per_scale.append(descriptors)
        del model
        torch.cuda.empty_cache()

    geometric = np.concatenate(per_scale, axis=1).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        query_points=points,
        geometric_features_30=per_scale[0],
        geometric_features_40=per_scale[1],
        geometric_features=geometric,
        scales=np.asarray(SCALES, dtype=np.float32),
        radii=np.asarray(radii, dtype=np.float32),
        seed=np.int32(args.seed),
        gedi_commit=np.array(GEDI_COMMIT),
    )

    print("points:", points.shape)
    print("geometric 30%:", per_scale[0].shape)
    print("geometric 40%:", per_scale[1].shape)
    print("geometric fused:", geometric.shape)
    print("finite:", bool(np.isfinite(geometric).all()))
    print("saved:", output)


if __name__ == "__main__":
    main()
