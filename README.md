# FreezeV2-Re

Minimal, paper-faithful reimplementation of **FreeZeV2 / FreeZeV2.1** for zero-shot 6D object pose estimation on the BOP benchmark.

The codebase is intentionally small. There is no pose-specific training or fine-tuning: DINOv2 and GeDi are used as frozen feature extractors. Segmentation is treated as an external input; the reproduction focuses on pose estimation.

## Current baseline

The `main` branch currently contains:

- BOP reference/evaluation harness
- deterministic Poisson-disk CAD surface sampling
- CNOS 162-view CAD onboarding cameras
- 480x480 RGB/depth template rendering with explicit 50% frame fill
- frozen DINOv2 intermediate-feature wrapper and FoundPose-style feature sampling
- multi-view query visual-feature aggregation with visibility filtering
- query-point visibility mapping and compressed onboarding cache
- RGB-D backprojection
- 16x16 sparse target sampling
- top-k cosine descriptor matching
- Kabsch rigid alignment
- feature-aware 3D-3D RANSAC
- BOP CSV submission writer

## Install / test

For the BOP/onboarding environment:

```bash
pip install -e '.[data,onboard,test]'
pytest -q
```

For the CUDA feature environment:

```bash
pip install -e '.[features,test]'
pytest tests/test_features.py -q
```

The project explicitly restricts setuptools discovery to `freezev2*`, so an `external/` directory such as `external/bop_toolkit` or `external/dinov2` does not become an accidental Python package.

## Stage 1: prepare BOP data and reproduce the public score

We start with LM-O because its BOP19 test subset is small enough for a fast end-to-end environment check. Training/PBR archives are intentionally not downloaded.

### 1. Pin the official BOP Toolkit

```bash
git clone https://github.com/thodan/bop_toolkit.git external/bop_toolkit
git -C external/bop_toolkit checkout cea62d651c7e395b2e1962b9749e4e89693c6ac4
pip install ./external/bop_toolkit
```

### 2. Download the LM-O evaluation assets

```bash
python run_bop.py prepare-data --dataset lmo --bop-root data/bop
```

This downloads/extracts only:

- `lmo_base.zip`
- `lmo_models.zip`
- `lmo_test_bop19.zip`

The expected dataset root is `data/bop/lmo/`, including `test_targets_bop19.json`, `models/`, and `test/`.

### 3. Download the authors' untouched FreeZeV2.1 result

```bash
python run_bop.py download-reference --dataset lmo --output-dir data/reference
```

The result CSV is external benchmark data and is ignored by git.

### 4. Run the official BOP19 localization evaluator

Configure the OpenGL/EGL renderer for the current machine before running the evaluator. `FreezeV2-Re` intentionally does not inject renderer-specific environment variables; it only sets `BOP_PATH` and otherwise inherits the caller environment. The GitHub Actions workflow configures software Mesa explicitly for its headless runner.

```bash
python run_bop.py evaluate-reference \
  --dataset lmo \
  --bop-root data/bop \
  --bop-toolkit external/bop_toolkit \
  --eval-root outputs/bop_eval
```

`run_bop.py` calls `scripts/eval_bop19_pose.py` from the pinned BOP Toolkit. It does not reimplement BOP metrics.

### LM-O reference target

The untouched FreeZeV2.1 public submission should reproduce:

| Metric | Official value |
| --- | ---: |
| AR | 0.771 |
| AR_VSD | 0.623 |
| AR_MSSD | 0.829 |
| AR_MSPD | 0.861 |
| Time / image | 29.805 s |

## Stage 2: CAD onboarding

FreeZeV2 renders 162 templates per object using the viewpoints proposed by CNOS. The implementation reproduces the CNOS 162-view icosphere geometry in NumPy and keeps the paper's render normalization explicit: square 480x480 renders with the projected model spanning approximately 50% of the frame.

Surface points are sampled with deterministic Poisson-disk spacing, matching the sampling property specified by the paper. The paper defines a raw point cloud `P_Q^raw`, retains only points visible in at least 18 rendered views, and reports 5k points for the final `P_Q`, but it does not publish `N_Q^raw`. Therefore Task 2 does not silently invent that raw count; the 5k sample below is a geometry/render smoke test. The final visibility filtering and feature aggregation are resolved when DINOv2 query features are added.

The camera-distance normalization is also an explicit reproduction choice because the paper specifies the final image occupancy but does not publish a unique focal-length/camera-distance pair.

Template rendering uses the pinned BOP Toolkit's `vispy` renderer rather than `pyrender`. This deliberately reuses the same headless EGL path already validated by the Stage-1 evaluator and avoids `pyrender`'s EGL-device enumeration, which can fail under software Mesa/llvmpipe with `Invalid device ID (0)`. Mesa/GLVND library paths remain machine-specific and are intentionally not hard-coded by this repository.

For a first real stop-gate check on LM-O object 1:

```bash
mkdir -p outputs/onboard/lmo_obj_000001/rgb

python - <<'PY'
from pathlib import Path
from PIL import Image
from freezev2.onboard import (
    load_mesh,
    make_template_cameras,
    render_templates,
    sample_query_points,
    save_onboarding_cache,
)

mesh_path = "data/bop/lmo/models/obj_000001.ply"
mesh = load_mesh(mesh_path)
query_points = sample_query_points(mesh, n=5000, seed=0)
cameras = make_template_cameras(n=162, size=480)
templates = render_templates(mesh_path, cameras, size=480, target_fill=0.5)

out = Path("outputs/onboard/lmo_obj_000001")
for i, template in enumerate(templates):
    Image.fromarray(template.rgb).save(out / "rgb" / f"{i:03d}.png")

save_onboarding_cache(
    out / "onboarding.npz",
    query_points,
    [template.camera for template in templates],
)
print("query_points:", query_points.shape)
print("templates:", len(templates))
print("cache:", out / "onboarding.npz")
PY
```

Expected structural output:

```text
query_points: (5000, 3)
templates: 162
```

Before adding DINOv2, inspect the 162 RGB renders and verify the object is centered, fully visible, and approximately half-frame across viewpoints.

## Stage 3: frozen DINOv2 visual features

FreeZeV2 specifies ViT-giant DINOv2 patch features from intermediate layers "as proposed in FoundPose", but does not publish the ViT-g layer index. FoundPose's public LM-O representation config uses DINOv2 `vits14-reg`, `facet=token`, `layer=9`, normalization enabled, and stride 14. We therefore keep the ViT-g `layer` mandatory and explicit instead of presenting an inferred layer as a paper constant.

To keep the DINO implementation reproducible, `DinoExtractor` pins the same DINOv2 source revision used as the FoundPose submodule:

```text
e1277af2ba9496fbadf7aec6eba56e8d882d1e35
```

The default backbone is the official `dinov2_vitg14`. Feature sampling follows FoundPose's exact image-coordinate convention: `grid_sample(..., align_corners=False)` after mapping `(x, y)` by `2 * point / (width, height) - 1`.

The paper explicitly renders 480x480 templates, while ViT-g/14 requires patch-aligned input dimensions. `DinoExtractor` keeps the paper's 480x480 render unchanged on disk and drops only the bottom/right remainder before DINO inference. Therefore a 480x480 template is consumed as 476x476 (`34 * 14`) and produces a 34x34 feature grid. This follows DINOv2's documented closest-smaller-multiple behavior rather than changing the paper's render resolution. The actual DINO crop size is exposed as `extractor.last_image_hw` / `extractor.compatible_image_hw(...)`, and points in the dropped border are excluded during multi-view aggregation.

For a stable local source checkout:

```bash
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout e1277af2ba9496fbadf7aec6eba56e8d882d1e35
```

Then, in the CUDA `freeze` environment, run a one-template smoke test. `layer=30` below is only one explicit candidate for the 40-block ViT-g backbone; it is **not** claimed as the final FreeZeV2 layer.

```bash
python - <<'PY'
import numpy as np
import torch
from PIL import Image
from freezev2.features import DinoExtractor

image = np.asarray(
    Image.open("outputs/onboard/lmo_obj_000001/rgb/000.png").convert("RGB")
)
extractor = DinoExtractor(
    device="cuda",
    layer=30,
    repo_or_dir="external/dinov2",
)
feature_map = extractor.encode(image)

print("DINO image hw:", extractor.last_image_hw)
print("feature map:", tuple(feature_map.shape))
print("dtype/device:", feature_map.dtype, feature_map.device)
print("finite:", bool(torch.isfinite(feature_map).all()))
print("frozen:", all(not p.requires_grad for p in extractor.model.parameters()))
PY
```

The structural target is:

```text
DINO image hw: (476, 476)
feature map: (1536, 34, 34)
finite: True
frozen: True
```

The query aggregation API is:

```python
feature_image_hws = [
    extractor.compatible_image_hw(template.depth.shape[:2])
    for template in templates
]
points, visual_features, view_counts = aggregate_query_visual_features(
    query_points,
    templates,
    feature_maps,
    depth_tolerance=...,
    min_views=18,
    view_weights=...,
    feature_image_hws=feature_image_hws,
)
```

The paper states that per-view DINO features are combined with a weighted average but does not specify the weighting rule. `view_weights` is therefore explicit; `None` uses a uniform average only for smoke tests. The function returns the raw weighted visual mean. Following Eq. (1), PCA and L2 normalization are applied in the next fusion stage, not here.

## Seven BOP Classic-Core reference targets

| Dataset | FreeZeV2.1 AR |
| --- | ---: |
| LM-O | 0.771 |
| T-LESS | 0.755 |
| TUD-L | 0.976 |
| IC-BIN | 0.697 |
| ITODD | 0.742 |
| HB | 0.892 |
| YCB-V | 0.915 |
| **AR_core** | **0.821** |

## Minimal pose API

```python
from freezev2.pipeline import estimate_pose_from_features

pose, score = estimate_pose_from_features(
    query_points,
    query_features,
    target_points,
    target_features,
    object_diameter,
    k=10,
    iterations=10_000,
)
```

`query_points/query_features` represent the onboarded CAD object; `target_points/target_features` are the sparse RGB-D scene representation.

## Reproduction policy

Paper-specified values are used directly when available. Parameters or implementation details omitted by the paper are kept explicit so they can be validated against the authors' public BOP submissions rather than silently guessed. Datasets, masks, weights, caches, official submissions, and evaluation outputs stay outside git.
