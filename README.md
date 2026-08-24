# FreezeV2-Re

Minimal, paper-faithful reimplementation of **FreeZeV2 / FreeZeV2.1** for zero-shot 6D object pose estimation on the BOP benchmark.

The codebase is intentionally small. There is no pose-specific training or fine-tuning: DINOv2 and GeDi are used as frozen feature extractors.

## Current baseline

The `reproduce-bop` branch currently contains the geometry/matching core needed after feature extraction:

- RGB-D backprojection
- 16x16 sparse target sampling
- top-k cosine descriptor matching
- Kabsch rigid alignment
- feature-aware 3D-3D RANSAC
- BOP CSV submission writer
- end-to-end synthetic pose recovery tests

The next implementation step is the paper-specific feature/onboarding path: 162 CAD views, DINOv2 ViT-G features, GeDi geometry features, descriptor fusion, ICP refinement, then FreeZeV2.1 SAR/multi-mask competition refinements.

## Install / test

```bash
pip install -e '.[test]'
pytest -q
```

## Minimal API

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

Paper-specified values are used directly when available. Parameters or implementation details omitted by the paper are kept explicit so they can be validated against the authors' public BOP submissions rather than silently guessed.
