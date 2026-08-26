from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fusion import VisualPCA, fuse_visual_geometric
from .geometry import backproject_depth
from .target_features import (
    TARGET_DINO_INPUT_SIZE,
    TARGET_DINO_MODE,
    TARGET_DINO_PATCH_GRID,
    direct_feature_map_tokens,
    resize_square_bbox_crop,
    target_patch_grid,
)


def extract_target_cache(args, *, dino_cls, gedi_cls) -> dict:
    """Build one FreeZeV2 target cache from an explicit candidate mask.

    The visual branch uses one direct DINO token per retained 16x16 square-bbox
    patch. The 224x224 crop size is a reverse-engineering choice implied by a
    16x16 token grid with ViT-g/14's 14-pixel patch size; the paper publishes
    the grid/direct-assignment rule, not this resize constant.
    """
    if args.scene_id < 0 or args.im_id < 0:
        raise ValueError("--scene-id and --im-id must be non-negative")
    if args.obj_id <= 0:
        raise ValueError("--obj-id must be positive")
    if int(args.grid_size) != TARGET_DINO_PATCH_GRID:
        raise ValueError(
            f"--grid-size must be {TARGET_DINO_PATCH_GRID} for direct target DINO tokens"
        )
    if args.dense_size <= 0:
        raise ValueError("--dense-size must be positive")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to load BOP RGB-D inputs") from exc

    stem = f"{args.dataset}_obj_{args.obj_id:06d}"
    query_cache = args.query_cache or Path("outputs/features") / f"{stem}_query.npz"
    scene_dir = (
        args.bop_root / args.dataset / args.split / f"{args.scene_id:06d}"
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
        raise FileNotFoundError(f"local DINOv2 checkout not found: {args.dinov2_root}")
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
            raise KeyError("query cache is missing: " + ", ".join(missing))
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
        raise ValueError("target extraction expects a 64D saved query PCA")
    if diameter <= 0 or not np.isfinite(diameter):
        raise ValueError("query object diameter must be positive and finite")
    if not np.isfinite(pca_mean).all() or not np.isfinite(pca_components).all():
        raise ValueError("query PCA state contains non-finite values")

    scene_camera = json.loads(scene_camera_path.read_text())
    camera_info = scene_camera.get(str(args.im_id))
    if camera_info is None:
        camera_info = scene_camera.get(f"{args.im_id:06d}")
    if camera_info is None:
        raise KeyError(f"image {args.im_id} missing from {scene_camera_path}")
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
    if len(dense_all) < int(args.dense_size):
        raise ValueError(
            f"mask has only {len(dense_all)} valid depth points; "
            f"need {args.dense_size} for the target support cloud"
        )
    rng = np.random.default_rng(args.seed)
    dense_ids = rng.choice(len(dense_all), size=int(args.dense_size), replace=False)
    dense_points = np.asarray(dense_all[dense_ids], dtype=np.float32)

    (
        sparse_image_xy,
        sparse_pixels,
        sparse_token_indices,
        square_bbox_xyxy,
    ) = target_patch_grid(mask, grid_size=args.grid_size)
    if len(sparse_pixels) == 0:
        raise ValueError("mask has no retained patch centers")

    sparse_depth = depth_mm[sparse_pixels[:, 1], sparse_pixels[:, 0]]
    valid_sparse_depth = np.isfinite(sparse_depth) & (sparse_depth > 0)
    sparse_image_xy = sparse_image_xy[valid_sparse_depth]
    sparse_pixels = sparse_pixels[valid_sparse_depth]
    sparse_token_indices = sparse_token_indices[valid_sparse_depth]
    if len(sparse_pixels) == 0:
        raise ValueError("mask has no retained patch centers with valid depth")
    if len(sparse_pixels) > TARGET_DINO_PATCH_GRID**2:
        raise RuntimeError("sparse grid returned more than 16x16 points")

    # Backproject the original-image patch centers. Raster/window centers use
    # integer cell boundaries, hence -0.5 converts to OpenCV pixel-center coords.
    opencv_xy = sparse_image_xy.astype(np.float64) - 0.5
    u = opencv_xy[:, 0]
    v = opencv_xy[:, 1]
    z = depth_mm[sparse_pixels[:, 1], sparse_pixels[:, 0]].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    sparse_points = np.stack((x, y, z), axis=1).astype(np.float32)

    dino = dino_cls(
        device=args.device,
        layer=dino_layer,
        facet=dino_facet,
        model_name=dino_model,
        repo_or_dir=args.dinov2_root,
    )
    target_crop = resize_square_bbox_crop(
        rgb,
        square_bbox_xyxy,
        output_size=TARGET_DINO_INPUT_SIZE,
    )
    feature_map = dino.encode(target_crop)
    dino_image_hw = getattr(dino, "last_image_hw", None)
    if dino_image_hw is None:
        dino_image_hw = (TARGET_DINO_INPUT_SIZE, TARGET_DINO_INPUT_SIZE)
    dino_image_hw = tuple(map(int, dino_image_hw))
    expected_dino_hw = (TARGET_DINO_INPUT_SIZE, TARGET_DINO_INPUT_SIZE)
    if dino_image_hw != expected_dino_hw:
        raise RuntimeError(
            f"target DINO consumed {dino_image_hw}, expected {expected_dino_hw}"
        )
    direct_tokens = direct_feature_map_tokens(
        feature_map,
        grid_size=TARGET_DINO_PATCH_GRID,
    )
    visual_features = np.asarray(
        direct_tokens[sparse_token_indices],
        dtype=np.float32,
    )
    del feature_map, direct_tokens
    if visual_features.shape != (len(sparse_pixels), len(pca_mean)):
        raise RuntimeError(
            "target DINO feature shape does not match the saved query PCA: "
            f"{visual_features.shape} vs (*, {len(pca_mean)})"
        )
    if not np.isfinite(visual_features).all():
        raise RuntimeError("target visual features contain non-finite values")

    gedi_extractor = gedi_cls(
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
        raise RuntimeError(f"unexpected target fused shape: {target_features.shape}")
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
        sparse_token_indices=np.asarray(sparse_token_indices, dtype=np.int32),
        sparse_points=sparse_points,
        dense_points=dense_points,
        visual_features=visual_features,
        geometric_features=geometric,
        target_features=target_features,
        square_bbox_xyxy=np.asarray(square_bbox_xyxy, dtype=np.float32),
        target_dino_mode=np.array(TARGET_DINO_MODE),
        target_dino_input_hw=np.asarray(expected_dino_hw, dtype=np.int32),
        target_dino_patch_grid=np.asarray(
            [TARGET_DINO_PATCH_GRID, TARGET_DINO_PATCH_GRID], dtype=np.int32
        ),
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
        sparse_sampling=np.array("smallest_square_bbox_patch_centers"),
        query_source=np.array(str(query_cache)),
        rgb_source=np.array(str(rgb_path)),
        depth_source=np.array(str(depth_path)),
        mask_source=np.array(str(args.mask)),
        scene_camera_source=np.array(str(scene_camera_path)),
    )

    return {
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
        "square_bbox_xyxy": [float(v) for v in square_bbox_xyxy],
        "target_dino_mode": TARGET_DINO_MODE,
        "target_dino_input_hw": list(expected_dino_hw),
        "target_dino_patch_grid": [
            TARGET_DINO_PATCH_GRID,
            TARGET_DINO_PATCH_GRID,
        ],
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
    }
