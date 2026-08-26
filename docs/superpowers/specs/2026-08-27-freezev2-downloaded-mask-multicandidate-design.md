# FreeZeV2 Downloaded-Mask Multi-Candidate Design

Date: 2026-08-27

## Goal

Add the pose-side multi-mask stage described by FreeZeV2 without integrating any segmentation model inference. Candidate masks are downloaded prediction artifacts and are treated as external inputs. The existing single-mask target extraction, coarse registration, ICP refinement, and Eq. (7) scoring stay unchanged and are reused per candidate.

## Source of masks

The initial supported source is the official BOP 2023 Task-4 default detections archive:

- `bop23_default_detections_for_task4.zip`
- LM-O payload: `cnos-fastsam_lmo-test.json`
- Produced by CNOS/FastSAM and also used as the default segmentation input by FoundPose.

Each detection record contains at least:

- `scene_id`
- `image_id`
- `category_id`
- `score`
- `bbox`
- `segmentation` with COCO-style RLE (`counts`, `size`)

The first implementation supports the uncompressed integer-list RLE format used by this official BOP artifact. A compressed string RLE is rejected explicitly instead of silently mis-decoding it.

The source abstraction is intentionally generic enough to add downloaded SAM-6D, NIDS, or MUSE prediction files later without modifying the pose pipeline.

## Commands

### `download-default-masks`

Downloads and extracts the official BOP Task-4 default detections into:

`data/detections/cnos-fastsam/cnos-fastsam_<dataset>-test.json`

The command is deterministic and skips a valid existing file unless `--force` is supplied.

### `estimate-multi-mask`

Inputs:

- dataset, scene id, image id, object id
- query cache
- one or more downloaded detection JSON sources
- localization instance count `N` resolved from `test_targets_bop19.json`
- candidate multiplier (`N+1` policy for FreeZeV2 baseline; exposed in metadata)
- translation-NMS threshold supplied explicitly because the paper does not publish the numerical threshold
- existing RANSAC/ICP parameters

Flow:

1. Load all detections matching `(scene_id, image_id, category_id=obj_id)` from each source.
2. Sort each source by segmentation confidence and retain at most `N+1` candidates for the baseline configuration.
3. Decode each retained RLE into a boolean mask in memory.
4. For each candidate mask, reuse the existing target extraction logic, coarse pose estimation, and ICP refinement logic without using the segmentation confidence in Eq. (7).
5. Rank refined candidate poses by the existing `final_score` from Eq. (7).
6. Apply translation-only NMS in descending `final_score` order.
7. Retain at most `N` distinct poses for localization output.
8. Save candidate provenance and the final selected pose list.

Segmentation confidence is used only for the pre-pose top-`N+1` candidate truncation required by the localization pipeline. It is not multiplied into or otherwise used by `final_score`.

## NMS

The paper states that duplicate refined poses are removed by NMS based on translation distance but does not publish the numerical threshold. Therefore the first implementation requires `--nms-translation-threshold-mm` rather than hiding a guessed constant. The chosen value is recorded in the output cache/report.

For two candidate poses `a` and `b`, candidate `b` is suppressed when:

`||t_a - t_b||_2 < threshold_mm`

Rotation is not used by NMS, matching the paper description.

## RLE decoding

COCO uncompressed RLE counts alternate background/foreground run lengths over the flattened mask in column-major (Fortran) order. Decoder requirements:

- `size == [H, W]`
- non-negative integer counts
- total count exactly `H*W`
- alternating 0/1 runs starting with background
- reshape with `order='F'`

Malformed detections are rejected with source/candidate context.

## Cache/report contract

The multi-mask result records, for every candidate:

- source name/path
- source detection index
- segmentation confidence
- target cache path
- coarse cache path
- fine cache path
- `coarse_score`
- `fine_feature_score`
- `icp_score`
- `final_score`
- refined `R`, `t`
- NMS selected/suppressed status

The final selected list is ordered by descending `final_score` after NMS.

Separately, the existing coarse cache contract is corrected so that `top1_similarity_mean` and `kth_similarity_mean` are saved into the `.npz`, matching the already printed JSON report. This does not change pose estimation behavior.

## Error handling

- Missing detection archive/file: explicit `FileNotFoundError`.
- No candidate for an object/image: report zero candidates instead of inventing a mask.
- Candidate mask with insufficient valid target depth: mark that candidate invalid and continue with the remaining masks; one bad segmentation must not abort the whole image.
- RANSAC/refinement failure for one candidate: record the failure and continue.
- If fewer than `N` valid distinct poses survive, return all surviving poses.

## Testing

TDD coverage will include:

1. COCO uncompressed RLE decodes to the expected binary mask and respects Fortran order.
2. Detection filtering matches scene/image/object and sorts by confidence.
3. Per-source localization truncation retains exactly top `N+1` when enough candidates exist.
4. Segmentation confidence is not used in final pose ordering; lower-confidence mask can win via higher `final_score`.
5. Translation NMS suppresses only candidates within the configured threshold and retains descending-score winners.
6. A bad candidate does not abort processing of other candidates.
7. Final output retains at most `N` poses.
8. Coarse `.npz` saves `top1_similarity_mean` and `kth_similarity_mean` consistently with `candidate_similarities`.
9. Existing single-mask and official BOP reference CI remain green.

## Out of scope for this change

- Running CNOS, SAM-6D, NIDS, or MUSE inference.
- Reverse-engineering an unpublished NMS threshold.
- Switching the default DINO backbone from `vitg14` to `vitg14_reg`.
- SAR / FreeZeV2-Accurate-specific refinement.
- Detection-mode unknown instance count handling beyond the current localization-focused path.
