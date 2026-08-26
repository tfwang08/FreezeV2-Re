import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


_RUN_BOP_PATH = Path(__file__).resolve().parents[1] / "run_bop.py"
_RUN_BOP_SPEC = importlib.util.spec_from_file_location("run_bop", _RUN_BOP_PATH)
assert _RUN_BOP_SPEC is not None and _RUN_BOP_SPEC.loader is not None
run_bop = importlib.util.module_from_spec(_RUN_BOP_SPEC)
_RUN_BOP_SPEC.loader.exec_module(run_bop)


def test_extract_query_gedi_reads_model_diameter_and_saves_64d(
    tmp_path,
    monkeypatch,
):
    bop_root = tmp_path / "data" / "bop"
    models = bop_root / "lmo" / "models"
    models.mkdir(parents=True)
    (models / "models_info.json").write_text(
        json.dumps({"1": {"diameter": 100.0}})
    )

    cache = tmp_path / "onboarding.npz"
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    np.savez_compressed(cache, query_points=points)
    output = tmp_path / "gedi.npz"
    calls = {}

    class FakeExtractor:
        def __init__(self, checkpoint, gedi_root, seed=0):
            calls["init"] = (Path(checkpoint), Path(gedi_root), seed)

        def encode(self, pts, cloud, object_diameter):
            calls["diameter"] = object_diameter
            np.testing.assert_array_equal(pts, points)
            np.testing.assert_array_equal(cloud, points)
            return np.concatenate([
                np.full((len(pts), 32), 30.0, dtype=np.float32),
                np.full((len(pts), 32), 40.0, dtype=np.float32),
            ], axis=1)

    monkeypatch.setattr(run_bop, "GediExtractor", FakeExtractor)
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "extract-query-gedi",
        "--dataset",
        "lmo",
        "--obj-id",
        "1",
        "--bop-root",
        str(bop_root),
        "--onboarding-cache",
        str(cache),
        "--gedi-root",
        str(tmp_path / "gedi"),
        "--checkpoint",
        str(tmp_path / "checkpoint.tar"),
        "--output",
        str(output),
        "--seed",
        "7",
    ])

    run_bop.main()

    assert calls["diameter"] == 100.0
    assert calls["init"][2] == 7
    with np.load(output, allow_pickle=False) as data:
        assert data["query_points"].shape == (5, 3)
        assert data["geometric_features_30"].shape == (5, 32)
        assert data["geometric_features_40"].shape == (5, 32)
        assert data["geometric_features"].shape == (5, 64)
        np.testing.assert_allclose(data["radii"], [30.0, 40.0])


def test_fuse_query_features_saves_128d_and_query_pca(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(80, 3)).astype(np.float32)
    visual = rng.normal(size=(80, 96)).astype(np.float32)
    geometric = rng.normal(size=(80, 64)).astype(np.float32)

    visual_path = tmp_path / "visual.npz"
    geometric_path = tmp_path / "gedi.npz"
    output = tmp_path / "query.npz"
    np.savez_compressed(
        visual_path,
        query_points=points,
        visual_features=visual,
    )
    np.savez_compressed(
        geometric_path,
        query_points=points,
        geometric_features=geometric,
        diameter=np.float32(100.0),
    )

    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "fuse-query-features",
        "--dataset",
        "lmo",
        "--obj-id",
        "1",
        "--visual",
        str(visual_path),
        "--geometric",
        str(geometric_path),
        "--output",
        str(output),
    ])

    run_bop.main()

    with np.load(output, allow_pickle=False) as data:
        fused = np.asarray(data["fused_features"], dtype=np.float32)
        assert data["query_points"].shape == (80, 3)
        assert fused.shape == (80, 128)
        assert data["pca_mean"].shape == (96,)
        assert data["pca_components"].shape == (64, 96)
        assert int(data["pca_dim"]) == 64
        assert float(data["diameter"]) == 100.0
        np.testing.assert_allclose(
            np.linalg.norm(fused[:, :64], axis=1),
            1.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(fused[:, 64:], axis=1),
            1.0,
            atol=1e-5,
        )
        assert np.isfinite(fused).all()


def test_extract_query_visual_defaults_to_freezv2_3d_to_2d_sampling(tmp_path, monkeypatch):
    points = np.arange(18, dtype=np.float32).reshape(6, 3)
    onboarding = tmp_path / "onboarding.npz"
    np.savez_compressed(onboarding, query_points=points)
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    dinov2_root = tmp_path / "dinov2"
    dinov2_root.mkdir()
    output = tmp_path / "visual.npz"
    templates = [object(), object()]
    calls = {}

    class FakeDinoExtractor:
        def __init__(
            self,
            device,
            layer,
            facet="token",
            model_name="dinov2_vitg14",
            repo_or_dir=None,
        ):
            calls["dino"] = (device, layer, facet, model_name, Path(repo_or_dir))

    def fake_load_templates(cache_path, images):
        calls["templates"] = (Path(cache_path), Path(images))
        return templates

    def fake_projection_aggregate(
        query_points,
        loaded_templates,
        extractor,
        depth_tolerance,
        min_views=18,
        view_weights=None,
        depth_sampling="inverse_bilinear",
    ):
        np.testing.assert_array_equal(query_points, points)
        assert loaded_templates is templates
        assert isinstance(extractor, FakeDinoExtractor)
        assert view_weights is None
        calls["aggregate"] = (depth_tolerance, min_views, depth_sampling)
        features = np.arange(len(points) * 12, dtype=np.float32).reshape(len(points), 12)
        counts = np.full(len(points), 21, dtype=np.int32)
        return points.copy(), features, counts

    def fail_pixel_lifting(*_args, **_kwargs):
        raise AssertionError("FreeZeV2 default query path must not use pixel lifting")

    monkeypatch.setattr(run_bop, "DinoExtractor", FakeDinoExtractor, raising=False)
    monkeypatch.setattr(
        run_bop,
        "load_onboarding_templates",
        fake_load_templates,
        raising=False,
    )
    monkeypatch.setattr(
        run_bop,
        "aggregate_query_visual_features_streaming",
        fake_projection_aggregate,
        raising=False,
    )
    monkeypatch.setattr(
        run_bop,
        "aggregate_query_visual_features_pixel_lifting_streaming",
        fail_pixel_lifting,
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "extract-query-visual",
        "--dataset",
        "lmo",
        "--obj-id",
        "1",
        "--layer",
        "30",
        "--depth-tolerance",
        "1.0",
        "--dino-model",
        "dinov2_vitg14_reg",
        "--onboarding-cache",
        str(onboarding),
        "--rgb-dir",
        str(rgb_dir),
        "--dinov2-root",
        str(dinov2_root),
        "--output",
        str(output),
    ])

    run_bop.main()

    assert calls["dino"] == (
        "cuda",
        30,
        "token",
        "dinov2_vitg14_reg",
        dinov2_root,
    )
    assert calls["templates"] == (onboarding, rgb_dir)
    assert calls["aggregate"] == (1.0, 18, "inverse_bilinear")
    with np.load(output, allow_pickle=False) as data:
        assert data["query_points"].shape == (6, 3)
        assert data["visual_features"].shape == (6, 12)
        np.testing.assert_array_equal(data["view_counts"], np.full(6, 21))
        assert str(np.asarray(data["visual_aggregation"]).item()) == (
            "freezv2_3d_to_2d_visible_view_uniform"
        )
        assert str(np.asarray(data["query_sampling_mode"]).item()) == "3d_to_2d_projection"
        assert str(np.asarray(data["query_visibility"]).item()) == "rendered_depth"
        assert int(data["dino_layer"]) == 30
        assert str(data["dino_facet"]) == "token"
        assert str(np.asarray(data["dino_model"]).item()) == "dinov2_vitg14_reg"
        assert float(data["depth_tolerance"]) == 1.0
        assert str(np.asarray(data["depth_sampling"]).item()) == "inverse_bilinear"
        assert int(data["min_views"]) == 18


# --- downloaded-mask multi-candidate regression tests ---

def test_decode_uncompressed_coco_rle_uses_fortran_order():
    segmentation = {"size": [2, 3], "counts": [1, 3, 2]}
    decoded = run_bop._decode_uncompressed_coco_rle(segmentation)
    expected = np.array(
        [[False, True, False], [True, True, False]],
        dtype=bool,
    )
    np.testing.assert_array_equal(decoded, expected)


def test_decode_uncompressed_coco_rle_rejects_compressed_counts():
    import pytest

    with pytest.raises(ValueError, match="compressed"):
        run_bop._decode_uncompressed_coco_rle(
            {"size": [2, 2], "counts": "encoded-rle"}
        )


def test_detection_filtering_sorts_and_truncates_per_source(tmp_path):
    detections = [
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 3,
            "score": 0.4,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 3,
            "score": 0.9,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 3,
            "score": 0.8,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 4,
            "score": 0.99,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
    ]
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(detections))

    selected = run_bop._load_detection_candidates(
        path,
        scene_id=1,
        im_id=2,
        obj_id=3,
        limit=2,
    )

    assert [item["source_detection_index"] for item in selected] == [1, 2]
    assert [item["segmentation_confidence"] for item in selected] == [0.9, 0.8]


def test_localization_instance_count_reads_bop_targets(tmp_path):
    path = tmp_path / "test_targets_bop19.json"
    path.write_text(json.dumps([
        {"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 2},
        {"scene_id": 1, "im_id": 2, "obj_id": 4, "inst_count": 1},
    ]))

    assert run_bop._load_localization_instance_count(
        path,
        scene_id=1,
        im_id=2,
        obj_id=3,
    ) == 2


def test_translation_nms_orders_by_final_score_not_segmentation_confidence():
    candidates = [
        {
            "final_score": 0.7,
            "segmentation_confidence": 0.99,
            "t_mm": [0.0, 0.0, 0.0],
        },
        {
            "final_score": 0.9,
            "segmentation_confidence": 0.10,
            "t_mm": [1.0, 0.0, 0.0],
        },
        {
            "final_score": 0.8,
            "segmentation_confidence": 0.50,
            "t_mm": [100.0, 0.0, 0.0],
        },
    ]

    selected, suppressed_by = run_bop._translation_nms(
        candidates,
        threshold_mm=5.0,
        max_count=2,
    )

    assert selected == [1, 2]
    assert suppressed_by[0] == 1
    assert suppressed_by[1] is None
    assert suppressed_by[2] is None


def test_download_default_masks_extracts_expected_member_and_skips_valid_file(
    tmp_path,
    monkeypatch,
):
    import io
    import zipfile

    payload = json.dumps([{
        "scene_id": 1,
        "image_id": 2,
        "category_id": 3,
        "score": 0.9,
        "segmentation": {"size": [2, 2], "counts": [0, 4]},
    }]).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "cnos-fastsam/cnos-fastsam_lmo_test.json",
            payload,
        )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return buffer.getvalue()

    calls = {"count": 0}

    def fake_urlopen(_url):
        calls["count"] += 1
        return FakeResponse()

    monkeypatch.setattr(run_bop.urllib.request, "urlopen", fake_urlopen)
    output_dir = tmp_path / "detections" / "cnos-fastsam"

    first = run_bop._download_default_masks("lmo", output_dir, force=False)
    second = run_bop._download_default_masks("lmo", output_dir, force=False)

    output = output_dir / "cnos-fastsam_lmo-test.json"
    assert output.is_file()
    assert json.loads(output.read_text())[0]["category_id"] == 3
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert calls["count"] == 1


def test_estimate_multi_mask_continues_bad_candidate_and_ranks_by_final_score(
    tmp_path,
    monkeypatch,
    capsys,
):
    bop_root = tmp_path / "bop"
    dataset_root = bop_root / "lmo"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test_targets_bop19.json").write_text(json.dumps([
        {"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 2},
    ]))
    detections = tmp_path / "detections.json"
    detections.write_text(json.dumps([
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 3,
            "score": 0.95,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 3,
            "score": 0.90,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
        {
            "scene_id": 1,
            "image_id": 2,
            "category_id": 3,
            "score": 0.10,
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
    ]))

    def fake_invoke(argv):
        command = argv[0]
        output = Path(argv[argv.index("--output") + 1])
        candidate_index = int(output.stem.split("_")[1])
        if command == "extract-target":
            if candidate_index == 1:
                raise ValueError("bad candidate")
            return {"output": str(output)}
        if command == "estimate-coarse-pose":
            return {
                "coarse_score": {0: 0.2, 2: 0.3}[candidate_index],
                "output": str(output),
            }
        if command == "refine-pose":
            final_score = {0: 0.4, 2: 0.9}[candidate_index]
            translation = {0: [0.0, 0.0, 0.0], 2: [100.0, 0.0, 0.0]}[
                candidate_index
            ]
            return {
                "coarse_feature_score": {0: 0.2, 2: 0.3}[candidate_index],
                "fine_feature_score": 0.8,
                "icp_score": 0.7,
                "final_score": final_score,
                "R": np.eye(3).tolist(),
                "t_mm": translation,
                "output": str(output),
            }
        raise AssertionError(command)

    monkeypatch.setattr(run_bop, "_invoke_main_command", fake_invoke)
    work_dir = tmp_path / "multi"
    output = work_dir / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "estimate-multi-mask",
        "--dataset",
        "lmo",
        "--scene-id",
        "1",
        "--im-id",
        "2",
        "--obj-id",
        "3",
        "--bop-root",
        str(bop_root),
        "--detection-json",
        str(detections),
        "--nms-translation-threshold-mm",
        "5",
        "--work-dir",
        str(work_dir),
        "--output",
        str(output),
    ])

    run_bop.main()
    report = json.loads(capsys.readouterr().out)

    assert report["instance_count"] == 2
    assert report["candidate_limit_per_source"] == 3
    assert report["candidate_count"] == 3
    assert report["valid_candidate_count"] == 2
    assert report["candidates"][1]["status"] == "invalid"
    assert "bad candidate" in report["candidates"][1]["error"]
    assert [item["candidate_index"] for item in report["selected"]] == [2, 0]
    assert report["selected"][0]["segmentation_confidence"] == 0.10
    assert report["selected_count"] == 2
    assert output.is_file()


def test_coarse_cache_persists_similarity_means(tmp_path, monkeypatch):
    query_cache = tmp_path / "query.npz"
    target_cache = tmp_path / "target.npz"
    output = tmp_path / "coarse.npz"
    query_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    np.savez_compressed(
        query_cache,
        query_points=query_points,
        fused_features=features,
        diameter=np.float32(100.0),
    )
    np.savez_compressed(
        target_cache,
        sparse_points=query_points.copy(),
        target_features=features.copy(),
        scene_id=np.int32(1),
        im_id=np.int32(2),
        obj_id=np.int32(3),
    )
    candidate_idx = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    candidate_sim = np.array(
        [[0.9, 0.4], [0.8, 0.3], [0.7, 0.2]],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        run_bop,
        "topk_cosine_matches",
        lambda *_args, **_kwargs: (candidate_idx, candidate_sim),
    )

    def fake_estimate(*_args, **_kwargs):
        return np.eye(4), 0.5, {
            "edge_similarity_threshold": 0.9,
            "winning_target_indices": np.array([0, 1, 2]),
            "winning_candidate_columns": np.array([0, 0, 0]),
            "winning_query_indices": np.array([0, 1, 2]),
            "inlier_count": 3,
            "inlier_target_count": 3,
            "degenerate_triplets": 0,
            "edge_pruned_triplets": 0,
            "valid_hypotheses": 1,
        }

    monkeypatch.setattr(run_bop, "estimate_pose_from_features", fake_estimate)
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "estimate-coarse-pose",
        "--dataset",
        "lmo",
        "--scene-id",
        "1",
        "--im-id",
        "2",
        "--obj-id",
        "3",
        "--query-cache",
        str(query_cache),
        "--target-cache",
        str(target_cache),
        "--top-k",
        "2",
        "--iterations",
        "1",
        "--output",
        str(output),
    ])

    run_bop.main()

    with np.load(output, allow_pickle=False) as data:
        assert float(data["top1_similarity_mean"]) == np.mean(candidate_sim[:, 0])
        assert float(data["kth_similarity_mean"]) == np.mean(candidate_sim[:, -1])
