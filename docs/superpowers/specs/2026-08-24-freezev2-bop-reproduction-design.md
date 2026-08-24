# FreeZeV2 BOP Reproduction Design

## Goal

Reproduce the FreeZeV2 / FreeZeV2.1 model-based 6D localization pipeline on the seven BOP-Classic-Core datasets, without any pose-specific training or fine-tuning, and progressively approach the BOP Challenge 2024 FreeZeV2.1 result of 82.1 AR_core.

## Scope

The first target is **BOP Challenge 2024 Track 1: model-based 6D localization of unseen objects**.

In scope:

- RGB-D inference.
- CAD-model onboarding.
- Frozen DINOv2 visual features.
- Frozen GeDi geometric features.
- Sparse target feature extraction.
- 3D-3D feature matching.
- Feature-aware RANSAC.
- ICP refinement.
- Feature-aware final pose scoring and NMS.
- Multi-mask ensemble.
- FreeZeV2.1 symmetry-aware refinement (SAR) and rendered-pose visual scoring.
- Official BOP evaluation on LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, and YCB-V.

Out of scope until Track 1 is reproduced:

- BOP 6D detection track.
- Training any network.
- Reimplementing segmentation networks themselves.
- Refactoring into a large framework.

## Reproduction policy

1. Paper-specified values are copied exactly when available.
2. DINOv2 and GeDi remain frozen and run under inference mode.
3. Details omitted by the paper are explicit parameters, never hidden constants.
4. Unknown details are resolved against the authors' public BOP submissions, not against BOP test ground truth.
5. The official BOP evaluator is the source of truth for benchmark scores.
6. Large datasets, masks, model weights, rendered templates, and public submission CSVs are not committed to Git.
7. The implementation stays small enough that `run_bop.py` and the files under `freezev2/` expose the whole algorithm.

## Paper / challenge constants to reproduce

The initial implementation should use the published values below:

- DINOv2 backbone: ViT-Giant / ViT-g/14 family, frozen.
- GeDi: frozen pretrained model.
- CAD render views: 162 per object.
- Render size: 480 x 480, with object scaled to occupy about 50% of image width/height.
- Query surface cloud: 5,000 points.
- Dense target cloud: 3,000 points.
- Sparse target cloud: at most 256 points from a 16 x 16 grid.
- Feature matching: top-k = 10.
- GeDi scale 1: 30% of object diameter, 32 dimensions.
- GeDi scale 2: 40% of object diameter, 32 dimensions.
- Final fused descriptor: 128 dimensions after PCA / normalization / fusion.
- RANSAC iterations: 10,000.
- RANSAC inlier threshold: 3% of object diameter.
- ICP correspondence threshold: 3% of object diameter.
- Base final score: product of coarse feature score, fine feature score, and ICP score with exponents alpha=beta=gamma=1.

The challenge configuration FreeZeV2.1 additionally uses SAR, more mask candidates (paper reports up to M=2N), and rendered-pose visual feature scoring.

## Important ambiguity to resolve experimentally

Public descriptions are not fully identical about the exact segmentation ensemble used by the winning submission: the FreeZeV2 method page lists SAM6D, NIDS, CNOS, and MUSE, while the BOP Challenge report describes the winning localization variant with SAM6D, NIDS, and CNOS. We will not guess. The reproduction will evaluate the candidate ensembles against the authors' public submissions and record which configuration best matches pose counts, ranking behavior, and final AR.

## Minimal code structure

The final Track-1 implementation should remain close to:

```text
FreezeV2-Re/
├── freezev2/
│   ├── __init__.py
│   ├── bop.py          # BOP loading, targets, submissions
│   ├── geometry.py     # RGB-D / projection / rigid geometry
│   ├── matching.py     # feature NN search
│   ├── features.py     # frozen DINOv2 + GeDi + PCA/fusion
│   ├── onboard.py      # CAD sampling, rendering, query representation
│   ├── pose.py         # Kabsch + feature-aware RANSAC
│   ├── refine.py       # ICP, scoring, SAR, NMS
│   └── pipeline.py     # one-object / one-mask pose flow
├── run_bop.py          # benchmark entry point
├── tests/
│   ├── test_core.py
│   ├── test_bop.py
│   ├── test_onboard.py
│   ├── test_features.py
│   └── test_refine.py
└── docs/
```

No factories, registries, nested configuration packages, or training framework are planned.

## Data flow

### Offline onboarding

```text
BOP CAD mesh
  -> deterministic 5k surface points
  -> 162 rendered RGB/depth views
  -> frozen DINOv2 visual features
  -> back-project / aggregate multi-view visual features to 5k query points
  -> frozen two-scale GeDi descriptors
  -> PCA + L2 normalization + fusion
  -> cached query points + 128D descriptors
```

### Online inference

```text
RGB + depth + candidate mask
  -> 16x16 mask-aware grid <= 256 sparse pixels
  -> back-project sparse points
  -> build 3k dense target cloud
  -> DINOv2 visual descriptors at sparse pixels
  -> two-scale GeDi descriptors at sparse 3D points
  -> same PCA / normalization / fusion
  -> top-10 query matches per target point
  -> feature-aware 10k-iteration RANSAC
  -> coarse pose
  -> ICP refinement
  -> coarse feature score * fine feature score * ICP score
  -> NMS / candidate ranking
  -> optional FreeZeV2.1 SAR and rendered-pose visual rescoring
  -> BOP CSV
```

## Validation strategy

We validate from easiest to hardest.

1. **Evaluator sanity:** download an author's public FreeZeV2.1 BOP submission and reproduce its official score with our local BOP toolkit installation.
2. **Pure geometry:** synthetic rigid transforms must be recovered by Kabsch/RANSAC with known inliers and injected outliers.
3. **Onboarding:** a synthetic mesh must produce exactly 5k query points and 162 render camera poses with valid visibility mappings.
4. **Frozen features:** DINOv2 and GeDi parameters must have `requires_grad=False`; feature dimensions, normalization, and PCA shapes are asserted.
5. **Single-image BOP debug:** save coarse/refined overlays and per-stage scores for selected LM-O images.
6. **Pose regression:** compare our per-image poses and scores with the authors' public submission before relying only on aggregate AR.
7. **Base benchmark:** reproduce the paper's FreeZeV2 configuration before adding competition-only refinements.
8. **Challenge benchmark:** add SAR / extra masks / rendered-pose scoring and target 82.1 AR_core.

## Acceptance gates

### Gate A — benchmark harness

The official public FreeZeV2.1 submission evaluated locally reproduces the official BOP score. No pose implementation work is considered benchmark-valid before this gate passes.

### Gate B — feature/onboarding pipeline

For a BOP object, onboarding produces a deterministic 5k-point query representation with 128D fused descriptors, and target extraction produces at most 256 sparse points with matching 128D descriptors.

### Gate C — paper FreeZeV2

The full base pipeline runs on all seven BOP-Classic-Core datasets and its aggregate AR is compared against the paper's base configurations, especially the 80.1 AR ensemble result.

### Gate D — FreeZeV2.1

SAR, the competition mask policy, and rendered-pose visual scoring are enabled. The final objective is 82.1 AR_core. If exact equality is blocked by unavailable upstream segmentation outputs or undocumented parameters, the remaining gap must be localized per dataset and per pipeline stage using the authors' public submissions.

## Debugging order when accuracy is low

Accuracy regressions are investigated in this order:

1. BOP coordinate conventions and units.
2. Mask / target instance selection.
3. DINOv2 crop, resize, patch indexing, layer/facet, and interpolation.
4. CAD render camera conventions and multi-view visual aggregation.
5. GeDi input scaling and two neighborhood radii.
6. PCA fitting and feature normalization.
7. top-k correspondence construction.
8. RANSAC sampling, pruning, inlier computation, and feature-aware fitness.
9. ICP implementation and score definition.
10. candidate score/ranking/NMS.
11. symmetry handling and SAR.

This order is intentional: later refinements cannot compensate for an incorrect representation or coordinate system.
