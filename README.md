# FreezeV2-Re

Minimal, paper-faithful reimplementation of **FreeZeV2 / FreeZeV2.1** for zero-shot 6D object pose estimation on the BOP benchmark.

The codebase is intentionally small. There is no pose-specific training or fine-tuning: DINOv2 and GeDi are used as frozen feature extractors. Segmentation is treated as an external input; the reproduction focuses on pose estimation.

## Current baseline

The `main` branch currently contains:

- BOP reference/evaluation harness
- deterministic Poisson-disk CAD surface sampling
- CNOS 162-view CAD onboarding cameras
- 480x480 RGB/depth template rendering with explicit 50% frame fill
- query-point visibility mapping and compressed onboarding cache
- RGB-D backprojection
- 16x16 sparse target sampling
- top-k cosine descriptor matching
- Kabsch rigid alignment
- feature-aware 3D-3D RANSAC
- BOP CSV submission writer

## Install / test

```bash
pip install -e '.[data,onboard,test]'
pytest -q
```

The project explicitly restricts setuptools discovery to `freezev2*`, so an `external/` directory such as `external/bop_toolkit` does not become an accidental Python package.

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

`run_bop.py` calls `scripts/eval_bop19_pose.py` from the pinned BOP Toolkit. It does not reimplement VSD/MSSD/MSPD.

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

`pyrender` needs its OpenGL backend selected before it is imported. On a headless EGL machine, set `PYOPENGL_PLATFORM=egl` in the shell before running onboarding. Mesa/GLVND library paths remain machine-specific and are intentionally not hard-coded by this repository. This differs from the pinned BOP VisPy renderer, which sets `PYOPENGL_PLATFORM=egl` internally.

For a first real stop-gate check on LM-O object 1:

```bash
export PYOPENGL_PLATFORM=egl
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

mesh = load_mesh("data/bop/lmo/models/obj_000001.ply")
query_points = sample_query_points(mesh, n=5000, seed=0)
cameras = make_template_cameras(n=162, size=480)
templates = render_templates(mesh, cameras, size=480, target_fill=0.5)

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
