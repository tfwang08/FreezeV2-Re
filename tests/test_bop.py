import pytest

from freezev2.bop import (
    BOP_TOOLKIT_COMMIT,
    REFERENCE_SUBMISSIONS,
    build_eval_command,
    load_bop_targets,
    validate_bop_result_rows,
)


def test_load_bop_targets_preserves_instance_count(tmp_path):
    path = tmp_path / "test_targets_bop19.json"
    path.write_text('[{"scene_id":1,"im_id":2,"obj_id":3,"inst_count":2}]')
    rows = load_bop_targets(path)
    assert rows == [{"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 2}]


def test_reference_manifest_contains_seven_localization_submissions():
    assert set(REFERENCE_SUBMISSIONS) == {"lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv"}
    assert REFERENCE_SUBMISSIONS["lmo"]["expected"]["ar"] == pytest.approx(0.771)
    assert REFERENCE_SUBMISSIONS["ycbv"]["expected"]["ar"] == pytest.approx(0.915)
    assert len(BOP_TOOLKIT_COMMIT) == 40


def test_validate_bop_result_rows_accepts_localization_row_without_unit_conversion():
    rows = [{
        "scene_id": "1",
        "im_id": "2",
        "obj_id": "3",
        "score": "0.75",
        "R": "1 0 0 0 1 0 0 0 1",
        "t": "10 20 30",
        "time": "1.25",
    }]
    validate_bop_result_rows(rows)
    assert rows[0]["t"] == "10 20 30"


def test_validate_bop_result_rows_rejects_bad_translation():
    rows = [{
        "scene_id": "1",
        "im_id": "2",
        "obj_id": "3",
        "score": "0.75",
        "R": "1 0 0 0 1 0 0 0 1",
        "t": "10 20",
        "time": "1.25",
    }]
    with pytest.raises(ValueError, match="translation"):
        validate_bop_result_rows(rows)


def test_build_eval_command_delegates_to_official_bop_toolkit(tmp_path):
    toolkit = tmp_path / "bop_toolkit"
    result = tmp_path / "results" / "freezev21_lmo-test.csv"
    eval_root = tmp_path / "eval"
    cmd, env = build_eval_command(
        toolkit,
        tmp_path / "bop",
        result,
        eval_root,
        num_workers=1,
    )
    assert str(toolkit / "scripts" / "eval_bop19_pose.py") in cmd
    assert "--renderer_type=vispy" in cmd
    assert "--result_filenames=freezev21_lmo-test.csv" in cmd
    assert f"--results_path={result.parent}" in cmd
    assert f"--eval_path={eval_root}" in cmd
    assert "--targets_filename=test_targets_bop19.json" in cmd
    assert env["BOP_PATH"] == str(tmp_path / "bop")


def test_bop_dataset_patterns_use_only_eval_assets():
    from freezev2.bop import bop_dataset_patterns

    patterns = bop_dataset_patterns("lmo")
    assert patterns == ["lmo_base.zip", "lmo_models.zip", "*test*bop19.zip"]
    assert not any("train" in pattern for pattern in patterns)


def test_bop_dataset_patterns_reject_unknown_dataset():
    from freezev2.bop import bop_dataset_patterns

    with pytest.raises(ValueError, match="dataset"):
        bop_dataset_patterns("unknown")
