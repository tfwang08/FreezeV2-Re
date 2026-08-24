# FreezeV2-Re

Minimal, paper-faithful reimplementation of **FreeZeV2 / FreeZeV2.1** for zero-shot 6D object pose estimation on the BOP benchmark.

The codebase is intentionally small. There is no pose-specific training or fine-tuning: DINOv2 and GeDi are used as frozen feature extractors. Segmentation is treated as an external input; the reproduction focuses on pose estimation.

## Current baseline

The `reproduce-bop` branch currently contains the geometry/matching core needed after feature extraction:

- RGB-D backprojection
- 16x16 sparse target sampling
- top-k cosine descriptor matching
- Kabsch rigid alignment
- feature-aware 3D-3D RANSAC
- BOP CSV submission writer
- end-to-end synthetic pose recovery tests

Stage 1 adds a reproducible BOP reference harness before any DINOv2/GeDi implementation. It downloads only the official BOP19 challenge test subset, downloads the authors' untouched FreeZeV2.1 submission, and delegates all VSD/MSSD/MSPD scoring to the official BOP Toolkit.

## Install / test

```bash
pip install -e '.[data,test]'
pytest -q
```

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

For a headless CUDA/Linux machine:

```bash
PYOPENGL_PLATFORM=egl python run_bop.py evaluate-reference \
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

This exact offline reproduction is the stop gate for Stage 1. We do not move on to DINOv2/GeDi until the evaluator agrees with the public leaderboard.

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
