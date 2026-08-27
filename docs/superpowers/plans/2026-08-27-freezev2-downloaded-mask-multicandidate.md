# FreeZeV2 Downloaded-Mask Multi-Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the downloaded CNOS/FastSAM multi-mask localization stage while reusing the existing single-mask target extraction, coarse RANSAC, ICP refinement, and Eq. (7) scoring paths.

**Architecture:** Keep production changes in `run_bop.py`, as requested. Add small same-file helpers for uncompressed COCO RLE decoding, detection filtering, target-count lookup, translation NMS, default-mask download, and recursive reuse of the existing CLI subcommands. `estimate-multi-mask` orchestrates one candidate at a time, records failures instead of aborting the image, ranks by existing `final_score`, and applies translation-only NMS.

**Tech Stack:** Python 3.11, NumPy, Pillow, argparse, stdlib urllib/zipfile, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-freezev2-downloaded-mask-multicandidate-design.md`

## Global Constraints

- Do not run CNOS, SAM-6D, NIDS, MUSE, or any segmentation model.
- Initial source is the official BOP 2023 Task-4 default detections archive.
- Support uncompressed integer-list COCO RLE; reject compressed string RLE explicitly.
- Per source, pre-truncate matching detections to the top `N+1` by segmentation confidence.
- Do not use segmentation confidence in Eq. (7) or final pose ordering.
- Translation NMS uses `||t_a - t_b||_2 < threshold_mm`; rotation is ignored.
- `--nms-translation-threshold-mm` is required because the paper does not publish the threshold.
- A bad candidate must not abort processing of other candidates.
- Keep existing single-mask commands and official BOP reference behavior unchanged.
- Production implementation stays in `run_bop.py`; no new business module.

---

### Task 1: RLE, candidate filtering, target count, and NMS helpers

**Files:**
- Modify: `run_bop.py`
- Test: `tests/test_run_bop.py`

**Interfaces:**
- Produces: `_decode_uncompressed_coco_rle(segmentation) -> np.ndarray`
- Produces: `_load_detection_candidates(path, *, scene_id, im_id, obj_id, limit) -> list[dict]`
- Produces: `_load_localization_instance_count(path, *, scene_id, im_id, obj_id) -> int`
- Produces: `_translation_nms(candidates, *, threshold_mm, max_count) -> tuple[list[int], dict[int, int | None]]`

- [ ] **Step 1: Write failing tests** for Fortran-order RLE decoding, compressed-RLE rejection, tuple filtering/confidence sorting/top-`N+1`, target-count lookup, and final-score-first translation NMS.
- [ ] **Step 2: Run** `pytest tests/test_run_bop.py -q` and verify failures are caused by the missing helpers.
- [ ] **Step 3: Implement minimal helpers** in `run_bop.py` with strict validation: `[H, W]`, non-negative integer runs, exact `H*W` coverage, and finite non-negative scores.
- [ ] **Step 4: Run** `pytest tests/test_run_bop.py -q` and verify the helper tests pass.

### Task 2: Default BOP Task-4 mask download command

**Files:**
- Modify: `run_bop.py`
- Test: `tests/test_run_bop.py`

**Interfaces:**
- Produces command: `download-default-masks`
- Default URL: `https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip`
- Output: `data/detections/cnos-fastsam/cnos-fastsam_<dataset>-test.json`

- [ ] **Step 1: Write a failing test** using an in-memory ZIP and monkeypatched `urlopen`; assert deterministic extraction and skip-without-`--force` behavior.
- [ ] **Step 2: Run** the targeted test and verify it fails because the command/helper is missing.
- [ ] **Step 3: Implement** stdlib-only download/extraction. Accept either archive member spelling `cnos-fastsam_<dataset>_test.json` or `cnos-fastsam_<dataset>-test.json`, and normalize the saved output name to the design contract.
- [ ] **Step 4: Re-run** the targeted test and `pytest tests/test_run_bop.py -q`.

### Task 3: Multi-mask orchestration command

**Files:**
- Modify: `run_bop.py`
- Test: `tests/test_run_bop.py`

**Interfaces:**
- Produces command: `estimate-multi-mask`
- Reuses existing commands: `extract-target`, `estimate-coarse-pose`, `refine-pose`
- Produces JSON report with per-candidate provenance/scores/status and `selected` pose list.

- [ ] **Step 1: Write failing orchestration tests** that monkeypatch only the expensive subcommand invoker. Cover: lower segmentation confidence winning on higher `final_score`; one candidate raising while another survives; at most `N` poses retained; nearby translations suppressed by NMS.
- [ ] **Step 2: Run** the targeted tests and verify expected failures.
- [ ] **Step 3: Implement `_invoke_main_command(argv)`** by saving/restoring `sys.argv` and capturing the nested command JSON output, so the new pipeline reuses current single-mask CLI behavior rather than duplicating RANSAC/ICP code.
- [ ] **Step 4: Add `estimate-multi-mask` arguments:** dataset/scene/image/object, one-or-more `--detection-json`, optional `--targets-json`, existing target/coarse/refine knobs, required NMS translation threshold, `--work-dir`, and optional final `--output`.
- [ ] **Step 5: Implement candidate loop:** decode RLE, write a candidate PNG, invoke existing target/coarse/refine commands with candidate-specific cache paths, record per-candidate failures, rank valid candidates only by `final_score`, apply translation NMS, and retain at most `N`.
- [ ] **Step 6: Save and print the JSON report** with source path/index, segmentation confidence, cache paths, coarse/fine/ICP/final scores, `R`, `t_mm`, and NMS selected/suppressed metadata.
- [ ] **Step 7: Run** `pytest tests/test_run_bop.py -q` and verify all new orchestration tests pass.

### Task 4: Coarse-cache metadata contract fix

**Files:**
- Modify: `run_bop.py`
- Test: `tests/test_pose_cli.py` or `tests/test_run_bop.py`

**Interfaces:**
- Existing coarse `.npz` additionally persists `top1_similarity_mean` and `kth_similarity_mean`.

- [ ] **Step 1: Write a failing assertion** against a coarse cache verifying both means equal the means derived from `candidate_similarities[:, 0]` and `candidate_similarities[:, -1]`.
- [ ] **Step 2: Run** the targeted test and verify the keys are currently missing.
- [ ] **Step 3: Add the two scalar fields** to the existing coarse cache payload without changing pose estimation.
- [ ] **Step 4: Re-run** the targeted test.

### Task 5: Regression verification

**Files:**
- Modify only if needed to correct regressions: `run_bop.py`, `tests/test_run_bop.py`

- [ ] **Step 1: Run** `pytest -q`.
- [ ] **Step 2: Open a PR against `main`** so `.github/workflows/bop-reference.yml` runs the full test suite and official LM-O reference evaluation.
- [ ] **Step 3: Verify** BOP19 AR remains within the existing CI tolerance of the public reference (`0.771 ± 5e-4`, with VSD/MSSD/MSPD checks unchanged).
- [ ] **Step 4: Review the final diff** for accidental segmentation-score use in final ranking, guessed NMS defaults, unrelated refactors, or changes to DINO/GeDi behavior.
