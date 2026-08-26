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
from freezev2.features import (
    DINOV2_FOUNDPOSE_COMMIT,
    DINOV2_MODEL_NAME,
    DinoExtractor,
    sample_feature_map,
)
from freezev2.fusion import VisualPCA, fit_visual_pca, fuse_visual_geometric
from freezev2.gedi_bridge import GEDI_REPO_COMMIT, GEDI_SCALES, GediExtractor
from freezev2.geometry import backproject_depth
from freezev2.matching import topk_cosine_matches
from freezev2.onboard import load_onboarding_templates
from freezev2.pipeline import estimate_pose_from_features
from freezev2.query_features import aggregate_query_visual_features_streaming


def _sample_mask_patch_centers(
    mask: np.ndarray,
    grid_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paper-style patch centers in the smallest square mask bbox.

    FreeZeV2 samples a ``grid_size x grid_size`` grid inside the smallest
    axis-aligned square bounding box enclosing the candidate mask, then keeps
    only patch centers that fall inside the mask. ``centers`` are expressed in
    raster/window coordinates, where pixel-cell boundaries are integers. The
    paired integer ``pixels`` select the depth cell containing each center.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape HxW")
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )

    x0 = float(xs.min())
    x1 = float(xs.max() + 1)
    y0 = float(ys.min())
    y1 = float(ys.max() + 1)
    side = max(x1 - x0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    square_x0 = cx - 0.5 * side
    square_y0 = cy - 0.5 * side
    step = side / grid_size

    axis = np.arange(grid_size, dtype=np.float64) + 0.5
    x_centers = square_x0 + axis * step
    y_centers = square_y0 + axis * step
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    centers = np.stack((xx.ravel(), yy.ravel()), axis=1)
    pixels = np.floor(centers).astype(np.int64)

    h, w = mask.shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    keep = np.zeros(len(centers), dtype=bool)
    ids = np.flatnonzero(inside)
    if len(ids):
        keep[ids] = mask[pixels[ids, 1], pixels[ids, 0]]
    return centers[keep].astype(np.float32), pixels[keep]


def _rotation_error_deg(predicted: np.ndarray, reference: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(3, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(3, 3)
    delta = predicted @ reference.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


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

    visual = subparsers.add_parser(
        "extract-query-visual",
        help="Aggregate frozen DINOv2 template features onto onboarded query points",
    )
    visual.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    visual.add_argument("--obj-id", type=int, required=True)
    visual.add_argument("--layer", type=int, required=True)
    visual.add_argument("--depth-tolerance", type=float, required=True)
    visual.add_argument("--device", default="cuda")
    visual.add_argument("--facet", default="token", choices=["token"])
    visual.add_argument("--min-views", type=int, default=18)
    visual.add_argument(
        "--depth-sampling",
        default="inverse_bilinear",
        choices=["nearest", "bilinear", "inverse_bilinear"],
    )
    visual.add_argument("--onboarding-cache", type=Path)
    visual.add_argument("--rgb-dir", type=Path)
    visual.add_argument("--dinov2-root", type=Path, default=Path("external/dinov2"))
    visual.add_argument("--output", type=Path)

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

    target = subparsers.add_parser(
        "extract-target",
        help="Build one sparse RGB-D target representation from an external mask",
    )
    target.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    target.add_argument("--scene-id", type=int, required=True)
    target.add_argument("--im-id", type=int, required=True)
    target.add_argument("--obj-id", type=int, required=True)
    target.add_argument("--mask", type=Path, required=True)
    target.add_argument("--bop-root", type=Path, default=Path("data/bop"))
    target.add_argument("--split", default="test")
    target.add_argument("--query-cache", type=Path)
    target.add_argument("--dinov2-root", type=Path, default=Path("external/dinov2"))
    target.add_argument("--gedi-root", type=Path, default=Path("external/gedi"))
    target.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/gedi/data/chkpts/3dmatch/chkpt.tar"),
    )
    target.add_argument("--device", default="cuda")
    target.add_argument("--grid-size", type=int, default=16)
    target.add_argument("--dense-size", type=int, default=3000)
    target.add_argument("--seed", type=int, default=0)
    target.add_argument("--output", type=Path)

    target_viz = subparsers.add_parser(
        "visualize-target",
        help="Overlay saved sparse target patch centers on their RGB/mask inputs",
    )
    target_viz.add_argument("--target-cache", type=Path, required=True)
    target_viz.add_argument("--output", type=Path)
    target_viz.add_argument("--radius", type=int, default=3)

    coarse = subparsers.add_parser(
        "estimate-coarse-pose",
        help="Match saved query/target descriptors and run feature-aware 3D-3D RANSAC",
    )
    coarse.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    coarse.add_argument("--scene-id", type=int, required=True)
    coarse.add_argument("--im-id", type=int, required=True)
    coarse.add_argument("--obj-id", type=int, required=True)
    coarse.add_argument("--query-cache", type=Path)
    coarse.add_argument("--target-cache", type=Path, required=True)
    coarse.add_argument("--bop-root", type=Path, default=Path("data/bop"))
    coarse.add_argument("--split", default="test")
    coarse.add_argument("--top-k", type=int, default=10)
    coarse.add_argument("--iterations", type=int, default=10_000)
    coarse.add_argument("--seed", type=int, default=0)
    coarse.add_argument(
        "--edge-tolerance",
        type=float,
        help="Triplet edge-length tolerance in 3D units; defaults to 0.03*diameter",
    )
    coarse.add_argument("--gt-id", type=int)
    coarse.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "prepare-data":
        print(prepare_bop_dataset(args.dataset, args.bop_root))
        return

    if args.command == "download-reference":
        print(download_reference_submission(args.dataset, args.output_dir))
        return

    if args.command == "estimate-coarse-pose":
        if args.scene_id < 0 or args.im_id < 0:
            raise ValueError("--scene-id and --im-id must be non-negative")
        if args.obj_id <= 0:
            raise ValueError("--obj-id must be positive")
        if args.top_k <= 0:
            raise ValueError("--top-k must be positive")
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive")
        if args.edge_tolerance is not None and args.edge_tolerance <= 0:
            raise ValueError("--edge-tolerance must be positive")
        if args.gt_id is not None and args.gt_id < 0:
            raise ValueError("--gt-id must be non-negative")

        query_cache = args.query_cache or (
            Path("outputs/features")
            / f"{args.dataset}_obj_{args.obj_id:06d}_query.npz"
        )
        if not query_cache.is_file():
            raise FileNotFoundError(f"query cache not found: {query_cache}")
        if not args.target_cache.is_file():
            raise FileNotFoundError(f"target cache not found: {args.target_cache}")

        with np.load(query_cache, allow_pickle=False) as data:
            required = ("query_points", "fused_features", "diameter")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError("query cache is missing: " + ", ".join(missing))
            query_points = np.asarray(data["query_points"], dtype=np.float32)
            query_features = np.asarray(data["fused_features"], dtype=np.float32)
            diameter = float(data["diameter"])

        with np.load(args.target_cache, allow_pickle=False) as data:
            required = ("sparse_points", "target_features")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError("target cache is missing: " + ", ".join(missing))
            target_points = np.asarray(data["sparse_points"], dtype=np.float32)
            target_features = np.asarray(data["target_features"], dtype=np.float32)
            for key, expected in (
                ("scene_id", args.scene_id),
                ("im_id", args.im_id),
                ("obj_id", args.obj_id),
            ):
                if key in data and int(data[key]) != int(expected):
                    raise ValueError(
                        f"target cache {key}={int(data[key])} does not match CLI {expected}"
                    )

        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError("query_points must have shape Nx3")
        if target_points.ndim != 2 or target_points.shape[1] != 3:
            raise ValueError("target sparse_points must have shape Nx3")
        if len(target_points) < 3:
            raise ValueError("coarse pose estimation requires at least 3 target points")
        if query_features.ndim != 2 or len(query_features) != len(query_points):
            raise ValueError("query fused_features must have shape NxD matching query_points")
        if target_features.ndim != 2 or len(target_features) != len(target_points):
            raise ValueError("target_features must have shape NxD matching sparse_points")
        if query_features.shape[1] != target_features.shape[1]:
            raise ValueError("query and target descriptor dimensions do not match")
        if args.top_k > len(query_points):
            raise ValueError("--top-k cannot exceed the number of query points")
        if diameter <= 0 or not np.isfinite(diameter):
            raise ValueError("query object diameter must be positive and finite")
        for name, array in (
            ("query_points", query_points),
            ("query_features", query_features),
            ("target_points", target_points),
            ("target_features", target_features),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")

        inlier_threshold = 0.03 * diameter
        edge_tolerance = (
            inlier_threshold
            if args.edge_tolerance is None
            else float(args.edge_tolerance)
        )
        candidate_idx, candidate_sim = topk_cosine_matches(
            target_features,
            query_features,
            k=args.top_k,
        )
        coarse_pose, coarse_score, ransac_debug = estimate_pose_from_features(
            query_points,
            query_features,
            target_points,
            target_features,
            diameter,
            k=args.top_k,
            iterations=args.iterations,
            seed=args.seed,
            edge_tolerance=edge_tolerance,
            return_debug=True,
        )
        coarse_pose = np.asarray(coarse_pose, dtype=np.float64)
        if coarse_pose.shape != (4, 4) or not np.isfinite(coarse_pose).all():
            raise RuntimeError("coarse pose must be a finite 4x4 matrix")
        if not np.isfinite(coarse_score):
            raise RuntimeError("RANSAC did not produce a finite coarse score")

        R_pred = coarse_pose[:3, :3]
        t_pred = coarse_pose[:3, 3]
        rotation_det = float(np.linalg.det(R_pred))
        rotation_orthogonality_error = float(
            np.linalg.norm(R_pred.T @ R_pred - np.eye(3), ord="fro")
        )

        gt_R = None
        gt_t = None
        rotation_error_deg = None
        translation_error_mm = None
        gt_path = None
        if args.gt_id is not None:
            gt_path = (
                args.bop_root
                / args.dataset
                / args.split
                / f"{args.scene_id:06d}"
                / "scene_gt.json"
            )
            if not gt_path.is_file():
                raise FileNotFoundError(f"scene GT not found: {gt_path}")
            scene_gt = json.loads(gt_path.read_text())
            annotations = scene_gt.get(str(args.im_id))
            if annotations is None:
                annotations = scene_gt.get(f"{args.im_id:06d}")
            if annotations is None:
                raise KeyError(f"image {args.im_id} missing from {gt_path}")
            if args.gt_id >= len(annotations):
                raise IndexError(
                    f"gt_id {args.gt_id} out of range for image {args.im_id}"
                )
            annotation = annotations[args.gt_id]
            if int(annotation["obj_id"]) != args.obj_id:
                raise ValueError(
                    f"GT entry obj_id={annotation['obj_id']} does not match --obj-id {args.obj_id}"
                )
            gt_R = np.asarray(annotation["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            gt_t = np.asarray(annotation["cam_t_m2c"], dtype=np.float64).reshape(3)
            rotation_error_deg = _rotation_error_deg(R_pred, gt_R)
            translation_error_mm = float(np.linalg.norm(t_pred - gt_t))

        output = args.output or (
            Path("outputs/poses")
            / (
                f"{args.dataset}_scene_{args.scene_id:06d}_im_{args.im_id:06d}_"
                f"obj_{args.obj_id:06d}_coarse.npz"
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coarse_pose": coarse_pose,
            "coarse_score": np.float64(coarse_score),
            "candidate_query_indices": np.asarray(candidate_idx, dtype=np.int64),
            "candidate_similarities": np.asarray(candidate_sim, dtype=np.float64),
            "diameter": np.float32(diameter),
            "inlier_threshold": np.float32(inlier_threshold),
            "edge_tolerance": np.float32(ransac_debug["edge_tolerance"]),
            "top_k": np.int32(args.top_k),
            "iterations": np.int32(args.iterations),
            "seed": np.int32(args.seed),
            "winning_target_indices": np.asarray(
                ransac_debug["winning_target_indices"], dtype=np.int64
            ),
            "winning_candidate_columns": np.asarray(
                ransac_debug["winning_candidate_columns"], dtype=np.int64
            ),
            "winning_query_indices": np.asarray(
                ransac_debug["winning_query_indices"], dtype=np.int64
            ),
            "inlier_count": np.int32(ransac_debug["inlier_count"]),
            "inlier_target_count": np.int32(ransac_debug["inlier_target_count"]),
            "degenerate_triplets": np.int32(ransac_debug["degenerate_triplets"]),
            "edge_pruned_triplets": np.int32(ransac_debug["edge_pruned_triplets"]),
            "valid_hypotheses": np.int32(ransac_debug["valid_hypotheses"]),
            "scene_id": np.int32(args.scene_id),
            "im_id": np.int32(args.im_id),
            "obj_id": np.int32(args.obj_id),
            "query_source": np.array(str(query_cache)),
            "target_source": np.array(str(args.target_cache)),
        }
        if args.gt_id is not None:
            payload.update({
                "gt_id": np.int32(args.gt_id),
                "gt_R": gt_R,
                "gt_t": gt_t,
                "rotation_error_deg": np.float64(rotation_error_deg),
                "translation_error_mm": np.float64(translation_error_mm),
                "scene_gt_source": np.array(str(gt_path)),
            })
        np.savez_compressed(output, **payload)

        top1 = np.asarray(candidate_sim[:, 0], dtype=np.float64)
        kth = np.asarray(candidate_sim[:, -1], dtype=np.float64)
        report = {
            "dataset": args.dataset,
            "scene_id": args.scene_id,
            "im_id": args.im_id,
            "obj_id": args.obj_id,
            "query_points": list(query_points.shape),
            "target_points": list(target_points.shape),
            "descriptor_dim": int(query_features.shape[1]),
            "top_k": args.top_k,
            "iterations": args.iterations,
            "inlier_threshold": float(inlier_threshold),
            "edge_tolerance": float(ransac_debug["edge_tolerance"]),
            "top1_similarity_range": [float(top1.min()), float(top1.max())],
            "top1_similarity_mean": float(top1.mean()),
            "kth_similarity_mean": float(kth.mean()),
            "coarse_score": float(coarse_score),
            "inlier_count": int(ransac_debug["inlier_count"]),
            "inlier_target_count": int(ransac_debug["inlier_target_count"]),
            "degenerate_triplets": int(ransac_debug["degenerate_triplets"]),
            "edge_pruned_triplets": int(ransac_debug["edge_pruned_triplets"]),
            "valid_hypotheses": int(ransac_debug["valid_hypotheses"]),
            "rotation_determinant": rotation_det,
            "rotation_orthogonality_error": rotation_orthogonality_error,
            "R": R_pred.tolist(),
            "t_mm": t_pred.tolist(),
            "output": str(output),
        }
        if args.gt_id is not None:
            report.update({
                "gt_id": args.gt_id,
                "rotation_error_deg": float(rotation_error_deg),
                "translation_error_mm": float(translation_error_mm),
            })
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "visualize-target":
        if args.radius <= 0:
            raise ValueError("--radius must be positive")
        if not args.target_cache.is_file():
            raise FileNotFoundError(f"target cache not found: {args.target_cache}")

        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("Install Pillow to visualize target caches") from exc

        required = ("sparse_image_xy", "sparse_points", "rgb_source", "mask_source")
        with np.load(args.target_cache, allow_pickle=False) as data:
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError("target cache is missing: " + ", ".join(missing))
            sparse_image_xy = np.asarray(data["sparse_image_xy"], dtype=np.float32)
            sparse_points = np.asarray(data["sparse_points"], dtype=np.float32)
            rgb_path = Path(str(np.asarray(data["rgb_source"]).item()))
            mask_path = Path(str(np.asarray(data["mask_source"]).item()))

        if sparse_image_xy.ndim != 2 or sparse_image_xy.shape[1] != 2:
            raise ValueError("sparse_image_xy must have shape Nx2")
        if sparse_points.shape != (len(sparse_image_xy), 3):
            raise ValueError("sparse_points must have shape Nx3 matching sparse_image_xy")
        if len(sparse_points) == 0:
            raise ValueError("target cache contains no sparse points")
        if not np.isfinite(sparse_image_xy).all() or not np.isfinite(sparse_points).all():
            raise ValueError("target cache contains non-finite sparse coordinates")
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB source not found: {rgb_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"mask source not found: {mask_path}")

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        mask_raw = np.asarray(Image.open(mask_path))
        if mask_raw.ndim == 3:
            mask = np.any(mask_raw != 0, axis=2)
        else:
            mask = mask_raw != 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError("RGB and mask source sizes do not match")

        overlay = rgb.astype(np.float32)
        tint = np.array([0.0, 255.0, 0.0], dtype=np.float32)
        overlay[mask] = 0.65 * overlay[mask] + 0.35 * tint
        overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(overlay_image)
        radius = float(args.radius)
        for x, y in sparse_image_xy:
            draw.ellipse(
                (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius),
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=1,
            )

        pixels = np.floor(sparse_image_xy).astype(np.int64)
        h, w = mask.shape
        inside_image = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        inside_mask = np.zeros(len(pixels), dtype=bool)
        ids = np.flatnonzero(inside_image)
        if len(ids):
            inside_mask[ids] = mask[pixels[ids, 1], pixels[ids, 0]]

        output = args.output or args.target_cache.with_name(
            f"{args.target_cache.stem}_overlay.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        overlay_image.save(output)

        xyz_min = sparse_points.min(axis=0)
        xyz_max = sparse_points.max(axis=0)
        print(json.dumps({
            "point_count": int(len(sparse_points)),
            "inside_image_count": int(inside_image.sum()),
            "inside_mask_count": int(inside_mask.sum()),
            "inside_mask_fraction": float(inside_mask.mean()),
            "xyz_min": [float(v) for v in xyz_min],
            "xyz_max": [float(v) for v in xyz_max],
            "z_range": [float(xyz_min[2]), float(xyz_max[2])],
            "rgb_source": str(rgb_path),
            "mask_source": str(mask_path),
            "target_cache": str(args.target_cache),
            "output": str(output),
        }, indent=2, sort_keys=True))
        return

    if args.command == "extract-target":
        if args.scene_id < 0 or args.im_id < 0:
            raise ValueError("--scene-id and --im-id must be non-negative")
        if args.obj_id <= 0:
            raise ValueError("--obj-id must be positive")
        if args.grid_size <= 0:
            raise ValueError("--grid-size must be positive")
        if args.dense_size <= 0:
            raise ValueError("--dense-size must be positive")

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install Pillow to load BOP RGB-D inputs") from exc

        stem = f"{args.dataset}_obj_{args.obj_id:06d}"
        query_cache = args.query_cache or (
            Path("outputs/features") / f"{stem}_query.npz"
        )
        scene_dir = (
            args.bop_root
            / args.dataset
            / args.split
            / f"{args.scene_id:06d}"
        )
        rgb_path = scene_dir / "rgb" / f"{args.im_id:06d}.png"
        depth_path = scene_dir / "depth" / f"{args.im_id:06d}.png"
        scene_camera_path = scene_dir / "scene_camera.json"

        for path, label in (
            (query_cache, "query cache"),
            (rgb_path, "RGB image"),
            (depth_path, "depth image"),
            (scene_camera_path, "scene camera JSON"),
            (args.mask, "mask"),
            (args.checkpoint, "GeDi checkpoint"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        if not args.dinov2_root.is_dir():
            raise FileNotFoundError(
                f"local DINOv2 checkout not found: {args.dinov2_root}"
            )
        if not args.gedi_root.is_dir():
            raise FileNotFoundError(f"GeDi checkout not found: {args.gedi_root}")

        required_query_keys = (
            "pca_mean",
            "pca_components",
            "pca_dim",
            "diameter",
            "dino_layer",
            "dino_facet",
            "dino_model",
        )
        with np.load(query_cache, allow_pickle=False) as data:
            missing = [key for key in required_query_keys if key not in data]
            if missing:
                raise KeyError(
                    "query cache is missing: " + ", ".join(missing)
                )
            pca_mean = np.asarray(data["pca_mean"], dtype=np.float64)
            pca_components = np.asarray(data["pca_components"], dtype=np.float64)
            pca_dim = int(data["pca_dim"])
            diameter = float(data["diameter"])
            dino_layer = int(data["dino_layer"])
            dino_facet = str(np.asarray(data["dino_facet"]).item())
            dino_model = str(np.asarray(data["dino_model"]).item())

        if pca_mean.ndim != 1 or pca_components.ndim != 2:
            raise ValueError("invalid PCA state in query cache")
        if pca_components.shape != (pca_dim, len(pca_mean)):
            raise ValueError("query PCA component shape is inconsistent")
        if pca_dim != 64:
            raise ValueError("Task 5 expects a 64D saved query PCA")
        if diameter <= 0:
            raise ValueError("query object diameter must be positive")
        if not np.isfinite(pca_mean).all() or not np.isfinite(pca_components).all():
            raise ValueError("query PCA state contains non-finite values")

        scene_camera = json.loads(scene_camera_path.read_text())
        camera_info = scene_camera.get(str(args.im_id))
        if camera_info is None:
            camera_info = scene_camera.get(f"{args.im_id:06d}")
        if camera_info is None:
            raise KeyError(
                f"image {args.im_id} missing from {scene_camera_path}"
            )
        if "cam_K" not in camera_info:
            raise KeyError(f"cam_K missing for image {args.im_id}")
        K = np.asarray(camera_info["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(camera_info.get("depth_scale", 1.0))
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        if not np.isfinite(K).all() or K[0, 0] == 0 or K[1, 1] == 0:
            raise ValueError("invalid camera intrinsic matrix")

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8).copy()
        depth_raw = np.asarray(Image.open(depth_path))
        mask_raw = np.asarray(Image.open(args.mask))
        if mask_raw.ndim == 3:
            mask = np.any(mask_raw != 0, axis=2)
        else:
            mask = mask_raw != 0
        if depth_raw.shape[:2] != rgb.shape[:2]:
            raise ValueError("RGB and depth images have different sizes")
        if mask.shape != rgb.shape[:2]:
            raise ValueError("mask and RGB image have different sizes")

        depth_mm = np.asarray(depth_raw, dtype=np.float32) * np.float32(depth_scale)
        valid_mask = mask & np.isfinite(depth_mm) & (depth_mm > 0)
        dense_all, _ = backproject_depth(depth_mm, K, valid_mask)
        if len(dense_all) == 0:
            raise ValueError("mask contains no valid depth pixels")
        rng = np.random.default_rng(args.seed)
        dense_count = min(int(args.dense_size), len(dense_all))
        dense_ids = rng.choice(len(dense_all), size=dense_count, replace=False)
        dense_points = np.asarray(dense_all[dense_ids], dtype=np.float32)

        dino = DinoExtractor(
            device=args.device,
            layer=dino_layer,
            facet=dino_facet,
            model_name=dino_model,
            repo_or_dir=args.dinov2_root,
        )
        crop_h, crop_w = dino.compatible_image_hw(rgb.shape[:2])

        sparse_image_xy, sparse_pixels = _sample_mask_patch_centers(
            mask,
            args.grid_size,
        )
        if len(sparse_pixels) == 0:
            raise ValueError("mask has no retained patch centers")

        inside_crop = (
            (sparse_image_xy[:, 0] >= 0.0)
            & (sparse_image_xy[:, 0] < crop_w)
            & (sparse_image_xy[:, 1] >= 0.0)
            & (sparse_image_xy[:, 1] < crop_h)
        )
        sparse_depth = depth_mm[sparse_pixels[:, 1], sparse_pixels[:, 0]]
        valid_sparse_depth = np.isfinite(sparse_depth) & (sparse_depth > 0)
        keep_sparse = inside_crop & valid_sparse_depth
        sparse_image_xy = sparse_image_xy[keep_sparse]
        sparse_pixels = sparse_pixels[keep_sparse]
        if len(sparse_pixels) == 0:
            raise ValueError(
                "mask has no valid sparse patch centers inside the DINO crop"
            )
        if len(sparse_pixels) > args.grid_size * args.grid_size:
            raise RuntimeError("sparse grid returned more than grid_size^2 points")

        opencv_xy = sparse_image_xy.astype(np.float64) - 0.5
        u = opencv_xy[:, 0]
        v = opencv_xy[:, 1]
        z = depth_mm[sparse_pixels[:, 1], sparse_pixels[:, 0]].astype(np.float64)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        sparse_points = np.stack((x, y, z), axis=1).astype(np.float32)

        feature_map = dino.encode(rgb)
        dino_image_hw = getattr(dino, "last_image_hw", None)
        if dino_image_hw is None:
            dino_image_hw = (crop_h, crop_w)
        dino_image_hw = tuple(map(int, dino_image_hw))
        sampled = sample_feature_map(
            feature_map,
            sparse_image_xy,
            image_hw=dino_image_hw,
        )
        visual_features = (
            sampled.detach().to("cpu").numpy().astype(np.float32, copy=False)
        )
        del feature_map
        if visual_features.shape != (len(sparse_pixels), len(pca_mean)):
            raise RuntimeError(
                "target DINO feature shape does not match the saved query PCA: "
                f"{visual_features.shape} vs (*, {len(pca_mean)})"
            )
        if not np.isfinite(visual_features).all():
            raise RuntimeError("target visual features contain non-finite values")

        gedi_extractor = GediExtractor(
            checkpoint=args.checkpoint,
            gedi_root=args.gedi_root,
            seed=args.seed,
        )
        geometric = np.asarray(
            gedi_extractor.encode(
                sparse_points,
                dense_points,
                object_diameter=diameter,
            ),
            dtype=np.float32,
        )
        if geometric.shape != (len(sparse_points), 64):
            raise RuntimeError(f"unexpected target GeDi shape: {geometric.shape}")
        if not np.isfinite(geometric).all():
            raise RuntimeError("target geometric features contain non-finite values")

        pca_state = VisualPCA(mean=pca_mean, components=pca_components)
        target_features = fuse_visual_geometric(
            visual_features,
            geometric,
            pca_state,
        )
        if target_features.shape != (len(sparse_points), 128):
            raise RuntimeError(
                f"unexpected target fused shape: {target_features.shape}"
            )
        if not np.isfinite(target_features).all():
            raise RuntimeError("target fused features contain non-finite values")

        visual_norms = np.linalg.norm(target_features[:, :64], axis=1)
        geometric_norms = np.linalg.norm(target_features[:, 64:], axis=1)
        output = args.output or (
            Path("outputs/targets")
            / (
                f"{args.dataset}_scene_{args.scene_id:06d}_im_{args.im_id:06d}_"
                f"obj_{args.obj_id:06d}_{args.mask.stem}.npz"
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            sparse_pixels=np.asarray(sparse_pixels, dtype=np.int32),
            sparse_image_xy=np.asarray(sparse_image_xy, dtype=np.float32),
            sparse_points=sparse_points,
            dense_points=dense_points,
            visual_features=visual_features,
            geometric_features=geometric,
            target_features=target_features,
            cam_K=K.astype(np.float32),
            depth_scale=np.float32(depth_scale),
            diameter=np.float32(diameter),
            scene_id=np.int32(args.scene_id),
            im_id=np.int32(args.im_id),
            obj_id=np.int32(args.obj_id),
            grid_size=np.int32(args.grid_size),
            dense_size_requested=np.int32(args.dense_size),
            dense_valid_count=np.int32(len(dense_all)),
            seed=np.int32(args.seed),
            dino_layer=np.int32(dino_layer),
            dino_facet=np.array(dino_facet),
            dino_model=np.array(dino_model),
            dino_image_hw=np.asarray(dino_image_hw, dtype=np.int32),
            sparse_sampling=np.array("square_bbox_patch_centers"),
            query_source=np.array(str(query_cache)),
            rgb_source=np.array(str(rgb_path)),
            depth_source=np.array(str(depth_path)),
            mask_source=np.array(str(args.mask)),
            scene_camera_source=np.array(str(scene_camera_path)),
        )
        print(json.dumps({
            "dataset": args.dataset,
            "scene_id": args.scene_id,
            "im_id": args.im_id,
            "obj_id": args.obj_id,
            "depth_scale": depth_scale,
            "valid_mask_depth_pixels": int(valid_mask.sum()),
            "sparse_image_xy": list(sparse_image_xy.shape),
            "sparse_pixels": list(sparse_pixels.shape),
            "sparse_points": list(sparse_points.shape),
            "dense_points": list(dense_points.shape),
            "visual_features": list(visual_features.shape),
            "geometric_features": list(geometric.shape),
            "target_features": list(target_features.shape),
            "visual_branch_norm_range": [
                float(visual_norms.min()),
                float(visual_norms.max()),
            ],
            "geometric_branch_norm_range": [
                float(geometric_norms.min()),
                float(geometric_norms.max()),
            ],
            "finite": bool(np.isfinite(target_features).all()),
            "output": str(output),
        }, indent=2, sort_keys=True))
        return

    if args.command == "extract-query-visual":
        if args.obj_id <= 0:
            raise ValueError("--obj-id must be positive")
        if args.layer < 0:
            raise ValueError("--layer must be non-negative")
        if args.depth_tolerance < 0:
            raise ValueError("--depth-tolerance must be non-negative")
        if args.min_views <= 0:
            raise ValueError("--min-views must be positive")

        stem = f"{args.dataset}_obj_{args.obj_id:06d}"
        onboard_dir = Path("outputs/onboard") / stem
        cache_path = args.onboarding_cache or (onboard_dir / "onboarding.npz")
        rgb_dir = args.rgb_dir or (onboard_dir / "rgb")
        output = args.output or (Path("outputs/features") / f"{stem}_visual.npz")

        if not cache_path.is_file():
            raise FileNotFoundError(f"onboarding cache not found: {cache_path}")
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"template RGB directory not found: {rgb_dir}")
        if not args.dinov2_root.is_dir():
            raise FileNotFoundError(
                f"local DINOv2 checkout not found: {args.dinov2_root}"
            )

        with np.load(cache_path, allow_pickle=False) as cache:
            if "query_points" not in cache:
                raise KeyError(f"query_points missing from {cache_path}")
            query_points = np.asarray(cache["query_points"], dtype=np.float32)
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError("onboarding query_points must have shape Nx3")

        templates = load_onboarding_templates(cache_path, rgb_dir)
        extractor = DinoExtractor(
            device=args.device,
            layer=args.layer,
            facet=args.facet,
            repo_or_dir=args.dinov2_root,
        )
        kept_points, visual_features, view_counts = (
            aggregate_query_visual_features_streaming(
                query_points,
                templates,
                extractor,
                depth_tolerance=args.depth_tolerance,
                min_views=args.min_views,
                view_weights=None,
                depth_sampling=args.depth_sampling,
            )
        )
        kept_points = np.asarray(kept_points, dtype=np.float32)
        visual_features = np.asarray(visual_features, dtype=np.float32)
        view_counts = np.asarray(view_counts, dtype=np.int32)

        if kept_points.ndim != 2 or kept_points.shape[1] != 3:
            raise RuntimeError(f"unexpected visual query-point shape: {kept_points.shape}")
        if visual_features.ndim != 2 or len(visual_features) != len(kept_points):
            raise RuntimeError(
                f"unexpected query visual-feature shape: {visual_features.shape}"
            )
        if view_counts.shape != (len(kept_points),):
            raise RuntimeError(f"unexpected view-count shape: {view_counts.shape}")
        if not np.isfinite(visual_features).all():
            raise RuntimeError("query visual features contain non-finite values")

        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            query_points=kept_points,
            visual_features=visual_features,
            view_counts=view_counts,
            dino_layer=np.int32(args.layer),
            dino_facet=np.array(args.facet),
            dino_model=np.array(DINOV2_MODEL_NAME),
            dino_commit=np.array(DINOV2_FOUNDPOSE_COMMIT),
            device=np.array(args.device),
            min_views=np.int32(args.min_views),
            depth_tolerance=np.float32(args.depth_tolerance),
            depth_sampling=np.array(args.depth_sampling),
            num_templates=np.int32(len(templates)),
            input_query_count=np.int32(len(query_points)),
            onboarding_source=np.array(str(cache_path)),
            rgb_source=np.array(str(rgb_dir)),
        )
        print(json.dumps({
            "dataset": args.dataset,
            "obj_id": args.obj_id,
            "dino_layer": args.layer,
            "dino_model": DINOV2_MODEL_NAME,
            "input_query_points": list(query_points.shape),
            "retained_query_points": list(kept_points.shape),
            "visual_features": list(visual_features.shape),
            "view_count_range": [
                int(view_counts.min()) if len(view_counts) else 0,
                int(view_counts.max()) if len(view_counts) else 0,
            ],
            "min_views": args.min_views,
            "depth_tolerance": args.depth_tolerance,
            "depth_sampling": args.depth_sampling,
            "finite": bool(np.isfinite(visual_features).all()),
            "output": str(output),
        }, indent=2, sort_keys=True))
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
            visual_features = np.asarray(data[args.visual_key], dtype=np.float32)
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
        if visual_features.ndim != 2 or geometric.ndim != 2:
            raise ValueError("visual and geometric features must have shape NxD")
        if len(visual_features) != len(points_visual) or len(geometric) != len(points_visual):
            raise ValueError("feature caches and query points have inconsistent lengths")
        if args.pca_dim != geometric.shape[1]:
            raise ValueError(
                "--pca-dim must equal the geometric feature dimension "
                f"({geometric.shape[1]})"
            )

        pca = fit_visual_pca(visual_features, output_dim=args.pca_dim)
        fused = fuse_visual_geometric(visual_features, geometric, pca)
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
            "visual_dim": np.int32(visual_features.shape[1]),
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
            "visual_features": list(visual_features.shape),
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
