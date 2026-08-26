from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from freezev2.bop import (
    BOP_TOOLKIT_COMMIT,
    REFERENCE_SUBMISSIONS,
    download_reference_submission,
    evaluate_reference,
    prepare_bop_dataset,
)
from freezev2.fusion import fit_visual_pca, fuse_visual_geometric
from freezev2.gedi_bridge import GEDI_REPO_COMMIT, GEDI_SCALES, GediExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeZeV2 BOP reproduction utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Download/extract the BOP19 challenge subset")
    prepare.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    prepare.add_argument("--bop-root", type=Path, default=Path("data/bop"))

    download = subparsers.add_parser("download-reference", help="Download the authors' public FreeZeV2.1 result CSV")
    download.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    download.add_argument("--output-dir", type=Path, default=Path("data/reference"))

    evaluate = subparsers.add_parser("evaluate-reference", help="Run the official BOP19 pose evaluator")
    evaluate.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    evaluate.add_argument("--bop-root", type=Path, default=Path("data/bop"))
    evaluate.add_argument("--result-csv", type=Path)
    evaluate.add_argument("--results-dir", type=Path, default=Path("data/reference"))
    evaluate.add_argument("--bop-toolkit", type=Path, default=Path("external/bop_toolkit"))
    evaluate.add_argument("--eval-root", type=Path, default=Path("outputs/bop_eval"))
    evaluate.add_argument("--num-workers", type=int, default=10)

    gedi = subparsers.add_parser(
        "extract-query-gedi",
        help="Extract two-scale GeDi descriptors for an onboarded CAD object",
    )
    gedi.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    gedi.add_argument("--obj-id", type=int, required=True)
    gedi.add_argument("--bop-root", type=Path, default=Path("data/bop"))
    gedi.add_argument("--onboarding-cache", type=Path)
    gedi.add_argument("--gedi-root", type=Path, default=Path("external/gedi"))
    gedi.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/gedi/data/chkpts/3dmatch/chkpt.tar"),
    )
    gedi.add_argument("--output", type=Path)
    gedi.add_argument("--seed", type=int, default=0)

    fuse = subparsers.add_parser(
        "fuse-query-features",
        help="Fit query-only visual PCA and fuse DINO/GeDi descriptors",
    )
    fuse.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    fuse.add_argument("--obj-id", type=int, required=True)
    fuse.add_argument("--visual", type=Path)
    fuse.add_argument("--geometric", type=Path)
    fuse.add_argument("--output", type=Path)
    fuse.add_argument("--visual-key", default="visual_features")
    fuse.add_argument("--geometric-key", default="geometric_features")
    fuse.add_argument("--points-key", default="query_points")
    fuse.add_argument("--pca-dim", type=int, default=64)

    args = parser.parse_args()

    if args.command == "prepare-data":
        print(prepare_bop_dataset(args.dataset, args.bop_root))
        return

    if args.command == "download-reference":
        print(download_reference_submission(args.dataset, args.output_dir))
        return

    if args.command == "extract-query-gedi":
        if args.obj_id <= 0:
            raise ValueError("--obj-id must be positive")

        models_info_path = (
            args.bop_root / args.dataset / "models" / "models_info.json"
        )
        models_info = json.loads(models_info_path.read_text())
        object_info = models_info.get(str(args.obj_id))
        if object_info is None or "diameter" not in object_info:
            raise KeyError(
                f"object {args.obj_id} has no diameter in {models_info_path}"
            )
        diameter = float(object_info["diameter"])
        if diameter <= 0:
            raise ValueError("object diameter must be positive")

        cache_path = args.onboarding_cache or (
            Path("outputs/onboard")
            / f"{args.dataset}_obj_{args.obj_id:06d}"
            / "onboarding.npz"
        )
        with np.load(cache_path, allow_pickle=False) as cache:
            if "query_points" not in cache:
                raise KeyError(f"query_points missing from {cache_path}")
            query_points = np.asarray(cache["query_points"], dtype=np.float32)

        extractor = GediExtractor(
            checkpoint=args.checkpoint,
            gedi_root=args.gedi_root,
            seed=args.seed,
        )
        geometric = extractor.encode(
            query_points,
            query_points,
            object_diameter=diameter,
        )
        if geometric.shape != (len(query_points), 64):
            raise RuntimeError(f"unexpected fused GeDi shape: {geometric.shape}")

        radii = np.asarray(
            [scale * diameter for scale in GEDI_SCALES],
            dtype=np.float32,
        )
        output = args.output or (
            Path("outputs/features")
            / f"{args.dataset}_obj_{args.obj_id:06d}_gedi.npz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            query_points=query_points,
            geometric_features_30=geometric[:, :32],
            geometric_features_40=geometric[:, 32:],
            geometric_features=geometric,
            scales=np.asarray(GEDI_SCALES, dtype=np.float32),
            radii=radii,
            diameter=np.float32(diameter),
            seed=np.int32(args.seed),
            gedi_commit=np.array(GEDI_REPO_COMMIT),
        )
        print(json.dumps({
            "dataset": args.dataset,
            "obj_id": args.obj_id,
            "diameter": diameter,
            "query_points": list(query_points.shape),
            "geometric_features": list(geometric.shape),
            "radii": radii.tolist(),
            "finite": bool(np.isfinite(geometric).all()),
            "output": str(output),
        }, indent=2, sort_keys=True))
        return

    if args.command == "fuse-query-features":
        if args.obj_id <= 0:
            raise ValueError("--obj-id must be positive")
        if args.pca_dim <= 0:
            raise ValueError("--pca-dim must be positive")

        stem = f"{args.dataset}_obj_{args.obj_id:06d}"
        visual_path = args.visual or (
            Path("outputs/features") / f"{stem}_visual.npz"
        )
        geometric_path = args.geometric or (
            Path("outputs/features") / f"{stem}_gedi.npz"
        )
        output = args.output or (
            Path("outputs/features") / f"{stem}_query.npz"
        )

        visual_metadata = {}
        with np.load(visual_path, allow_pickle=False) as data:
            if args.points_key not in data:
                raise KeyError(f"{args.points_key} missing from {visual_path}")
            if args.visual_key not in data:
                raise KeyError(f"{args.visual_key} missing from {visual_path}")
            points_visual = np.asarray(data[args.points_key], dtype=np.float32)
            visual = np.asarray(data[args.visual_key], dtype=np.float32)
            for key in (
                "view_counts",
                "dino_layer",
                "dino_facet",
                "dino_model",
                "dino_commit",
                "min_views",
                "depth_tolerance",
                "depth_sampling",
            ):
                if key in data:
                    visual_metadata[key] = np.array(data[key], copy=True)

        geometric_metadata = {}
        with np.load(geometric_path, allow_pickle=False) as data:
            if args.points_key not in data:
                raise KeyError(f"{args.points_key} missing from {geometric_path}")
            if args.geometric_key not in data:
                raise KeyError(f"{args.geometric_key} missing from {geometric_path}")
            points_geometric = np.asarray(data[args.points_key], dtype=np.float32)
            geometric = np.asarray(data[args.geometric_key], dtype=np.float32)
            for key in ("diameter", "scales", "radii", "seed", "gedi_commit"):
                if key in data:
                    geometric_metadata[key] = np.array(data[key], copy=True)

        if points_visual.ndim != 2 or points_visual.shape[1] != 3:
            raise ValueError("visual query_points must have shape Nx3")
        if points_geometric.ndim != 2 or points_geometric.shape[1] != 3:
            raise ValueError("geometric query_points must have shape Nx3")
        if points_visual.shape != points_geometric.shape or not np.allclose(
            points_visual,
            points_geometric,
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "visual and geometric caches do not describe the same points"
            )
        if visual.ndim != 2 or geometric.ndim != 2:
            raise ValueError("visual and geometric features must have shape NxD")
        if len(visual) != len(points_visual) or len(geometric) != len(points_visual):
            raise ValueError("feature caches and query points have inconsistent lengths")
        if args.pca_dim != geometric.shape[1]:
            raise ValueError(
                "--pca-dim must equal the geometric feature dimension "
                f"({geometric.shape[1]})"
            )

        pca = fit_visual_pca(visual, output_dim=args.pca_dim)
        fused = fuse_visual_geometric(visual, geometric, pca)
        expected_shape = (len(points_visual), args.pca_dim + geometric.shape[1])
        if fused.shape != expected_shape:
            raise RuntimeError(f"unexpected fused query shape: {fused.shape}")
        if not np.isfinite(fused).all():
            raise RuntimeError("fused query features contain non-finite values")

        visual_norms = np.linalg.norm(fused[:, : args.pca_dim], axis=1)
        geometric_norms = np.linalg.norm(fused[:, args.pca_dim :], axis=1)
        payload = {
            "query_points": points_visual,
            "fused_features": fused,
            "pca_mean": pca.mean.astype(np.float32),
            "pca_components": pca.components.astype(np.float32),
            "visual_dim": np.int32(visual.shape[1]),
            "geometric_dim": np.int32(geometric.shape[1]),
            "pca_dim": np.int32(args.pca_dim),
            "visual_source": np.array(str(visual_path)),
            "geometric_source": np.array(str(geometric_path)),
        }
        payload.update(visual_metadata)
        payload.update(geometric_metadata)

        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
        print(json.dumps({
            "dataset": args.dataset,
            "obj_id": args.obj_id,
            "query_points": list(points_visual.shape),
            "visual_features": list(visual.shape),
            "geometric_features": list(geometric.shape),
            "pca_components": list(pca.components.shape),
            "fused_features": list(fused.shape),
            "visual_branch_norm_range": [
                float(visual_norms.min()),
                float(visual_norms.max()),
            ],
            "geometric_branch_norm_range": [
                float(geometric_norms.min()),
                float(geometric_norms.max()),
            ],
            "finite": bool(np.isfinite(fused).all()),
            "output": str(output),
        }, indent=2, sort_keys=True))
        return

    spec = REFERENCE_SUBMISSIONS[args.dataset]
    result_csv = args.result_csv or (args.results_dir / spec["filename"])
    score_path = evaluate_reference(
        dataset=args.dataset,
        bop_root=args.bop_root,
        result_csv=result_csv,
        bop_toolkit=args.bop_toolkit,
        eval_root=args.eval_root,
        num_workers=args.num_workers,
    )
    scores = json.loads(score_path.read_text())
    print(json.dumps({
        "dataset": args.dataset,
        "scores": scores,
        "expected": spec["expected"],
        "bop_toolkit_commit": BOP_TOOLKIT_COMMIT,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
