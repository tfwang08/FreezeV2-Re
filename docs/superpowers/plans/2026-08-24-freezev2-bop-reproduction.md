# FreeZeV2 BOP Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce FreeZeV2 first, then the BOP Challenge 2024 FreeZeV2.1 configuration, with a small training-free codebase and official BOP evaluation.

**Architecture:** Keep the current small `freezev2/` package. Build one benchmark path from BOP CAD/RGB-D/masks to query and target fused descriptors, feature-aware 3D registration, ICP/scoring, and BOP CSV. Only after the paper FreeZeV2 path is benchmarked do we add FreeZeV2.1 SAR, extra-mask policy, and rendered-pose visual rescoring.

**Tech Stack:** Python 3.10+, PyTorch/CUDA, NumPy, Open3D or equivalent ICP backend, DINOv2 ViT-g/14 frozen backbone, GeDi frozen descriptor model, official BOP Toolkit.

**Spec:** `docs/superpowers/specs/2026-08-24-freezev2-bop-reproduction-design.md`

## Global Constraints

- No pose-specific training or fine-tuning.
- DINOv2 and GeDi parameters stay frozen.
- Target BOP task is 2024 Track 1: model-based 6D localization of unseen objects.
- Seven core datasets: LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V.
- Use official BOP evaluator; do not implement a private approximation of AR.
- Do not tune against BOP test ground truth.
- Paper-specified constants are copied exactly; omitted details remain explicit parameters.
- Do not commit datasets, masks, model weights, template caches, or public submission CSVs.
- Keep production code close to nine small modules plus `run_bop.py`; do not introduce framework abstractions.
- Implement one task at a time. Do not start the next task until its acceptance test passes.

---

## Milestone map

1. **Reference harness:** prove our BOP environment reproduces the authors' published submission score.
2. **CAD onboarding:** 5k points + 162 rendered views with valid geometry mappings.
3. **Frozen DINOv2:** paper-style visual descriptors.
4. **Frozen GeDi + fusion:** two geometric scales and 128D fused descriptors.
5. **Online target extraction:** 16x16 sparse points + 3k dense target cloud.
6. **Registration hardening:** make the existing feature-aware RANSAC paper-faithful.
7. **ICP + scoring/NMS:** finish base FreeZeV2 pose selection.
8. **Base BOP reproduction:** first LM-O, then all seven datasets, target the paper FreeZeV2 results including 80.1 AR ensemble.
9. **FreeZeV2.1:** SAR + M=2N mask policy + rendered-pose visual rescoring.
10. **Challenge reproduction:** all seven datasets, target 82.1 AR_core and localize any remaining gap.

---

### Task 1: Lock the BOP reference/evaluation harness

**Why first:** If the evaluator, units, target list, or CSV format is wrong, every later accuracy number is meaningless.

**Files:**
- Modify: `freezev2/bop.py`
- Create: `run_bop.py`
- Create: `tests/test_bop.py`
- Create: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: official BOP dataset roots and an external result CSV path.
- Produces: `load_bop_targets(path) -> list[dict]`, `validate_bop_result_rows(rows) -> None`, and a `run_bop.py evaluate-reference` command that delegates scoring to the official BOP Toolkit.

- [ ] **Step 1: Write a failing target-list test**

```python
from freezev2.bop import load_bop_targets


def test_load_bop_targets_preserves_instance_count(tmp_path):
    path = tmp_path / "test_targets_bop19.json"
    path.write_text('[{"scene_id":1,"im_id":2,"obj_id":3,"inst_count":2}]')
    rows = load_bop_targets(path)
    assert rows == [{"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 2}]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/test_bop.py::test_load_bop_targets_preserves_instance_count -q`

Expected: failure because `load_bop_targets` does not yet exist.

- [ ] **Step 3: Implement only JSON target loading and result validation**

`freezev2/bop.py` should reject missing BOP localization fields and preserve translations in millimetres exactly as stored in submission rows.

- [ ] **Step 4: Add `run_bop.py evaluate-reference`**

The command accepts `--bop-root`, `--dataset`, `--result-csv`, and `--bop-toolkit`. It invokes the official `eval_bop19_pose.py`; it does not reimplement VSD/MSSD/MSPD.

- [ ] **Step 5: Verify an authors' public FreeZeV2.1 submission**

Run the official evaluator first on LM-O, then one second dataset. Record the command and resulting official score in `README.md`.

Acceptance: local evaluation of the untouched public submission matches the corresponding BOP public score to evaluator precision.

- [ ] **Step 6: Protect external assets**

`.gitignore` must include at least `data/`, `weights/`, `outputs/`, `cache/`, `*.csv`, and Python build/test caches.

- [ ] **Step 7: Run all tests and commit**

Run: `pytest -q`

Commit: `feat: add BOP reference evaluation harness`

**Stop gate:** Do not implement model features until the public submission can be evaluated correctly.

---

### Task 2: Reproduce CAD onboarding geometry and 162 template views

**Files:**
- Create: `freezev2/onboard.py`
- Extend: `freezev2/geometry.py`
- Create: `tests/test_onboard.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: BOP mesh path and object diameter.
- Produces:
  - `sample_query_points(mesh, n=5000, seed=0) -> np.ndarray[5000,3]`
  - `make_template_cameras(n=162) -> list[CameraPose]`
  - `render_templates(mesh, cameras, size=480) -> list[Template]`
  - `map_visible_pixels_to_query_points(...)` for later multi-view feature aggregation.

- [ ] **Step 1: Write failing deterministic surface-sampling test**

```python
def test_query_sampling_is_exact_and_deterministic(cube_mesh):
    a = sample_query_points(cube_mesh, n=5000, seed=0)
    b = sample_query_points(cube_mesh, n=5000, seed=0)
    assert a.shape == (5000, 3)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Verify RED, then implement area-weighted triangle sampling**

No learning library should be involved in this step.

- [ ] **Step 3: Write failing 162-camera test**

```python
def test_template_camera_count_and_rotation_validity():
    cams = make_template_cameras(n=162)
    assert len(cams) == 162
    for cam in cams:
        assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-5)
        assert np.linalg.det(cam.R) > 0.999
```

- [ ] **Step 4: Implement the template camera distribution and render normalization**

Use the same 162-view spherical template convention as the BOP/FoundPose onboarding family. Render at 480x480 and compute camera distance so the model occupies approximately 50% of image width/height, as specified by FreeZeV2.

- [ ] **Step 5: Add a visibility/back-projection test**

For a rendered cube, visible depth pixels projected through the stored camera must land on the rendered surface within a small numeric tolerance.

- [ ] **Step 6: Save an onboarding cache format**

Use a single compressed `.npz` per object containing query points, camera matrices, and later descriptors. Do not create a database layer.

- [ ] **Step 7: Run tests and commit**

Run: `pytest tests/test_onboard.py tests/test_core.py -q`

Commit: `feat: add CAD onboarding geometry and templates`

**Stop gate:** inspect one LM-O object's 162 renders and query points before adding DINOv2.

---

### Task 3: Add frozen DINOv2 visual feature extraction

**Files:**
- Create: `freezev2/features.py`
- Create: `tests/test_features.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `DinoExtractor(device, layer, facet="token")`
  - `DinoExtractor.encode(image) -> torch.Tensor[C,Hf,Wf]`
  - `sample_feature_map(feature_map, pixels_xy, image_hw) -> torch.Tensor[N,C]`
  - `aggregate_query_visual_features(...) -> np.ndarray[5000,C]`

- [ ] **Step 1: Write a failing bilinear feature-sampling test**

Use a synthetic feature map whose value encodes its x/y coordinates; assert exact sampling at patch centres and interpolated values between centres.

- [ ] **Step 2: Implement feature-map sampling independent of DINOv2**

This isolates the most error-prone pixel/patch coordinate convention from model loading.

- [ ] **Step 3: Add a frozen-backbone integration test**

The test is marked `@pytest.mark.gpu` and asserts:

```python
assert all(not p.requires_grad for p in extractor.model.parameters())
assert extractor.model.training is False
```

- [ ] **Step 4: Implement DINOv2 ViT-g/14 loading under inference mode**

Use the official pretrained backbone, ImageNet normalization used by DINOv2/FoundPose, and patch stride 14 initially. Expose `layer` and `facet` as explicit constructor values rather than hiding them.

- [ ] **Step 5: Resolve the paper's "as in FoundPose" intermediate-layer ambiguity explicitly**

FoundPose publicly uses token features at an intermediate layer for its own backbone. For ViT-g/14, benchmark the explicit candidate layer indices `{29, 30, 31, 39}` with `facet="token"`; store the selected value in one constant only after pose-regression evidence is available. Do not use BOP test GT for this selection.

- [ ] **Step 6: Aggregate rendered-view visual descriptors to the 5k query points**

Each query point receives descriptors from views in which it is visible; aggregate only valid visible observations and L2-normalize the result.

- [ ] **Step 7: Run CPU tests, then one GPU smoke test, and commit**

Run: `pytest tests/test_features.py -q`

GPU smoke: one rendered LM-O template through ViT-g/14 with shape and finite-value assertions.

Commit: `feat: add frozen DINOv2 visual descriptors`

**Stop gate:** visualize nearest-neighbour DINO correspondences between two rendered views of the same CAD object.

---

### Task 4: Add frozen two-scale GeDi and 128D feature fusion

**Files:**
- Modify: `freezev2/features.py`
- Extend: `tests/test_features.py`

**Interfaces:**
- Produces:
  - `GediExtractor(checkpoint, device)`
  - `encode_gedi(points, cloud, diameter, radii=(0.30,0.40)) -> np.ndarray[N,64]`
  - `fit_visual_pca(query_visual, out_dim=64) -> PCAState`
  - `fuse_features(visual, geometric, pca_state) -> np.ndarray[N,128]`

- [ ] **Step 1: Write failing feature-fusion shape/norm tests**

```python
def test_fused_features_are_128d_and_normalized():
    fused = fuse_features(visual, geometric, pca_state)
    assert fused.shape == (len(visual), 128)
    assert np.isfinite(fused).all()
```

Also assert the visual branch is 64D after PCA and the geometric branch is 64D from two 32D GeDi scales.

- [ ] **Step 2: Wrap the official frozen GeDi `compute(pts, pcd)` API**

Do not fork the GeDi model architecture into this repository. Keep its dependency behind a small adapter.

- [ ] **Step 3: Implement the exact two relative radii**

Scale neighborhoods by `0.30 * object_diameter` and `0.40 * object_diameter` and concatenate their 32D outputs.

- [ ] **Step 4: Fit PCA only from the onboarded query representation**

The target uses the saved query PCA transform; target/test data never fit or update PCA.

- [ ] **Step 5: Add frozen-model assertions and one CAD smoke test**

Verify all GeDi parameters are frozen and one LM-O object produces finite 64D geometric descriptors at 5k query points.

- [ ] **Step 6: Update `.npz` onboarding cache**

Cache query points, fused 128D descriptors, PCA state, diameter, and the paper/reverse-engineered configuration needed to reproduce them.

- [ ] **Step 7: Run tests and commit**

Run: `pytest tests/test_features.py tests/test_onboard.py -q`

Commit: `feat: add GeDi and fused query representation`

**Stop gate:** for one CAD object, query representation must be exactly `[5000, 128]` and deterministic for fixed inputs.

---

### Task 5: Complete online RGB-D target feature extraction

**Files:**
- Modify: `freezev2/features.py`
- Modify: `freezev2/geometry.py`
- Modify: `freezev2/pipeline.py`
- Extend: `tests/test_features.py`

**Interfaces:**
- Produces `extract_target(rgb, depth, mask, K, diameter, pca_state, ...) -> TargetRepresentation` with:
  - `sparse_points`: `[N,3]`, `N <= 256`
  - `dense_points`: `[3000,3]` when enough valid depth exists
  - `features`: `[N,128]`
  - `pixels`: `[N,2]`

- [ ] **Step 1: Strengthen the existing 16x16 sparse sampling test**

Cover masks touching image borders, holes in depth, empty grid cells, and deterministic ordering.

- [ ] **Step 2: Add a 3k dense-cloud test**

The target dense cloud is sampled only from valid masked depth pixels and is deterministic under a fixed seed.

- [ ] **Step 3: Extract DINO descriptors exactly at the retained sparse pixels**

Do not compute target descriptors for all masked depth pixels.

- [ ] **Step 4: Compute two-scale GeDi descriptors using the 3k dense cloud as neighborhood support**

The descriptor centres are the sparse target points, not all dense points.

- [ ] **Step 5: Apply the saved query PCA and fuse to 128D**

Assert that no fitting occurs online.

- [ ] **Step 6: Add a real BOP crop smoke test**

For one LM-O RGB-D image and one known mask, assert finite points/descriptors and save a debug overlay of the <=256 sampled pixels.

- [ ] **Step 7: Run tests and commit**

Commit: `feat: add sparse RGB-D target descriptors`

**Stop gate:** visually verify sparse pixels and 3D points on several LM-O masks before touching RANSAC again.

---

### Task 6: Harden the existing feature-aware RANSAC to paper behavior

**Files:**
- Modify: `freezev2/pose.py`
- Modify: `freezev2/matching.py`
- Extend: `tests/test_core.py`

**Interfaces:**
- Keeps `estimate_pose_from_features(...)` stable where practical.
- Ensures top-k=10 matching and feature-aware hypothesis fitness are exactly separated from standard geometric inlier ratio.

- [ ] **Step 1: Add failing tests for top-k ambiguity**

Construct correspondences where the correct query match is not top-1 but is within top-10; the feature-aware RANSAC must recover the ground-truth rigid transform.

- [ ] **Step 2: Add failing triplet-degeneracy tests**

Reject collinear source triplets and correspondence triplets with incompatible pairwise edge lengths before Kabsch.

- [ ] **Step 3: Preserve paper constants**

Default `k=10`, `iterations=10000`, `inlier_threshold=0.03 * diameter`.

- [ ] **Step 4: Implement batched/GPU hypothesis generation only after NumPy/Torch reference tests agree**

The CPU/reference implementation remains useful for correctness; the GPU path must match its selected pose on seeded synthetic tests.

- [ ] **Step 5: Add pose-regression diagnostics**

Return optional debug data: winning correspondence indices, inlier count, coarse feature score, and random seed.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_core.py -q`

Commit: `feat: align feature-aware RANSAC with FreeZeV2`

**Stop gate:** synthetic outlier-heavy tests must recover rotation/translation within the predefined numeric tolerance on both reference and GPU paths.

---

### Task 7: Add ICP, final feature score, candidate ranking, and NMS

**Files:**
- Create: `freezev2/refine.py`
- Modify: `freezev2/pipeline.py`
- Create: `tests/test_refine.py`

**Interfaces:**
- Produces:
  - `refine_icp(query_points, target_dense, coarse_pose, threshold) -> (fine_pose, icp_score)`
  - `feature_pose_score(...) -> float`
  - `final_pose_score(coarse_feat, fine_feat, icp, alpha=1,beta=1,gamma=1) -> float`
  - `translation_nms(candidates, threshold) -> list[PoseCandidate]`

- [ ] **Step 1: Write a failing ICP convergence test**

Perturb a known synthetic pose slightly; ICP must reduce point alignment error.

- [ ] **Step 2: Implement ICP with max correspondence distance `0.03 * diameter`**

Keep iteration count and convergence tolerances explicit parameters because the paper does not fully specify them.

- [ ] **Step 3: Write exact final-score test**

```python
def test_default_final_score_is_product():
    assert final_pose_score(0.5, 0.4, 0.8) == pytest.approx(0.16)
```

- [ ] **Step 4: Recompute feature score after ICP**

The fine feature score must use the refined transform, not reuse the coarse inlier set blindly.

- [ ] **Step 5: Add deterministic candidate ranking and translation NMS**

NMS threshold remains one named reverse-engineering parameter and is logged with each benchmark run.

- [ ] **Step 6: End-to-end synthetic test**

`query representation -> matching -> RANSAC -> ICP -> score` must improve or preserve alignment relative to coarse pose.

- [ ] **Step 7: Run tests and commit**

Commit: `feat: add FreeZeV2 refinement and scoring`

**Stop gate:** base pose pipeline is now complete; do not add SAR yet.

---

### Task 8: Reproduce base FreeZeV2 on BOP before competition refinements

**Files:**
- Modify: `run_bop.py`
- Modify: `freezev2/bop.py`
- Modify: `README.md`

**Interfaces:**
- `run_bop.py onboard --dataset lmo ...`
- `run_bop.py infer --dataset lmo --masks ...`
- `run_bop.py evaluate --dataset lmo --result ...`

- [ ] **Step 1: Run exactly one LM-O target image**

Save a debug bundle containing RGB overlay, mask, sampled sparse points, coarse pose, fine pose, and component scores.

- [ ] **Step 2: Compare that image with the authors' public submission**

Measure rotation/translation difference for matching object instances and determine the earliest pipeline stage that explains a large difference.

- [ ] **Step 3: Run the full LM-O localization target list**

Use one segmentation source first. Verify every required scene/image/object has the expected number of returned candidates before evaluating AR.

- [ ] **Step 4: Fix representation/geometry errors before hyperparameter exploration**

Do not compensate for wrong DINO indexing, units, or camera frames by changing RANSAC/ICP thresholds.

- [ ] **Step 5: Extend to all seven datasets with the same implementation**

Dataset-specific code is limited to BOP paths/model choice (not algorithm forks). Use CAD models for T-LESS as required by the official FreeZeV2.1 description.

- [ ] **Step 6: Reproduce the paper base ensemble**

Process the documented segmentation-source candidates with base FreeZeV2 scoring and compare aggregate AR to the paper's 80.1 base-ensemble result.

- [ ] **Step 7: Record a table in `README.md`**

Columns: dataset, paper/reference AR, reproduced AR, difference, runtime, mask source, commit SHA, configuration hash.

- [ ] **Step 8: Commit**

Commit: `bench: reproduce base FreeZeV2 on BOP`

**Stop gate:** only when the base pipeline's remaining error is localized do we add challenge-only refinements.

---

### Task 9: Add FreeZeV2.1 SAR and competition scoring

**Files:**
- Modify: `freezev2/refine.py`
- Modify: `freezev2/onboard.py`
- Modify: `freezev2/pipeline.py`
- Extend: `tests/test_refine.py`

**Interfaces:**
- Produces:
  - geometric symmetry candidates estimated during onboarding using Chamfer distance
  - `sar_refine(...) -> PoseCandidate`
  - `rendered_visual_score(...) -> float`

- [ ] **Step 1: Add symmetry-candidate tests on synthetic symmetric objects**

Use a cylinder/box with known discrete or continuous symmetry and verify equivalent transforms are discovered/accepted while clearly non-symmetric transforms are rejected.

- [ ] **Step 2: Estimate and cache geometric symmetries during onboarding**

This matches the official FreeZeV2.1 method description: symmetry estimation belongs to onboarding, not test-time learning.

- [ ] **Step 3: Implement SAR as a separate refinement after base pose estimation**

Each symmetry-related candidate is rendered and compared with input visual evidence; keep base pose unchanged if no candidate improves the competition score.

- [ ] **Step 4: Implement rendered-pose visual scoring**

Use the same frozen DINO visual representation family; no learned head is introduced.

- [ ] **Step 5: Implement the paper's Accurate candidate budget**

For localization, allow up to `M = 2N` masks per segmentation source/configuration as described for FreeZeV2-Accurate.

- [ ] **Step 6: Resolve 3-source vs 4-source winning ensemble deterministically**

Run both documented candidate sets:

- SAM6D + NIDS + CNOS
- SAM6D + NIDS + CNOS + MUSE

Compare candidate counts, per-dataset score, and pose similarity to the authors' public submissions. Record the selected reproduction setting and evidence in `README.md`.

- [ ] **Step 7: Run tests and commit**

Commit: `feat: add FreeZeV2.1 symmetry-aware refinement`

---

### Task 10: Full BOP Challenge 2024 reproduction and gap report

**Files:**
- Modify: `README.md`
- Create: `docs/bop_reproduction_results.md`

**Interfaces:**
- No new algorithm API. This task freezes the benchmark configuration and reports evidence.

- [ ] **Step 1: Freeze one configuration**

Record DINO version/layer/facet, GeDi checkpoint, PCA behavior, seeds, RANSAC/ICP parameters, mask sources, SAR settings, dependency versions, GPU, and commit SHA.

- [ ] **Step 2: Run all seven BOP-Classic-Core datasets**

Generate one BOP CSV per dataset with no manual edits.

- [ ] **Step 3: Evaluate with the official BOP Toolkit**

Target aggregate: **82.1 AR_core** for the challenge FreeZeV2.1 configuration.

- [ ] **Step 4: Compare per dataset with the authors' public submissions**

For any meaningful gap, classify it as one of: segmentation/mask candidates, visual representation, geometric representation, coarse registration, ICP, score/ranking, or symmetry/SAR.

- [ ] **Step 5: Report reproducibility honestly**

`docs/bop_reproduction_results.md` must include exact commands, hardware, runtime, per-dataset AR, AR_core, known undocumented choices, and differences from the published submission.

- [ ] **Step 6: Run final regression suite**

Run: `pytest -q`

Also rerun one known LM-O debug image and the public-submission evaluator sanity check from Task 1.

- [ ] **Step 7: Only then mark the reproduction milestone complete**

Commit: `bench: report FreeZeV2.1 BOP reproduction`

---

## Execution rule for this project

We will **not** implement this whole plan in one pass. We execute one numbered task at a time. After each task:

1. run its tests / benchmark gate;
2. show the result;
3. inspect the small diff;
4. only then move to the next task.

The immediate next task is **Task 1: BOP reference/evaluation harness**. It is intentionally independent of DINOv2 and GeDi so the benchmark foundation is correct before any model work starts.
