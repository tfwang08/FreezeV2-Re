import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


_RUN_BOP_PATH = Path(__file__).resolve().parents[1] / "run_bop.py"
_RUN_BOP_SPEC = importlib.util.spec_from_file_location("run_bop_localization", _RUN_BOP_PATH)
assert _RUN_BOP_SPEC is not None and _RUN_BOP_SPEC.loader is not None
run_bop = importlib.util.module_from_spec(_RUN_BOP_SPEC)
_RUN_BOP_SPEC.loader.exec_module(run_bop)


def test_run_localization_groups_by_image_and_writes_bop_csv(
    tmp_path,
    monkeypatch,
    capsys,
):
    bop_root = tmp_path / "bop"
    dataset_root = bop_root / "lmo"
    dataset_root.mkdir(parents=True)
    targets = dataset_root / "test_targets_bop19.json"
    targets.write_text(json.dumps([
        {"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 1},
        {"scene_id": 1, "im_id": 2, "obj_id": 4, "inst_count": 1},
        {"scene_id": 1, "im_id": 3, "obj_id": 3, "inst_count": 1},
    ]))

    query_cache_dir = tmp_path / "features"
    query_cache_dir.mkdir()
    for obj_id in (3, 4):
        (query_cache_dir / f"lmo_obj_{obj_id:06d}_query.npz").write_bytes(b"cache")

    detection_a = tmp_path / "a.json"
    detection_b = tmp_path / "b.json"
    output = tmp_path / "freezev2-re_lmo-test.csv"
    work_dir = tmp_path / "localization"
    calls = []

    def fake_invoke(argv):
        argv = list(map(str, argv))
        calls.append(argv)
        assert argv[0] == "estimate-multi-mask"
        scene_id = int(argv[argv.index("--scene-id") + 1])
        im_id = int(argv[argv.index("--im-id") + 1])
        obj_id = int(argv[argv.index("--obj-id") + 1])
        assert (scene_id, im_id) == (1, 2)
        assert argv.count("--detection-json") == 2
        query_cache = Path(argv[argv.index("--query-cache") + 1])
        assert query_cache == query_cache_dir / f"lmo_obj_{obj_id:06d}_query.npz"
        return {
            "selected": [{
                "final_score": 1.0 - 0.1 * (obj_id - 3),
                "R": np.eye(3).tolist(),
                "t_mm": [float(obj_id), 20.0, 30.0],
            }],
        }

    monkeypatch.setattr(run_bop, "_invoke_main_command", fake_invoke)
    monkeypatch.setattr(sys, "argv", [
        "run_bop.py",
        "run-localization",
        "--dataset",
        "lmo",
        "--bop-root",
        str(bop_root),
        "--query-cache-dir",
        str(query_cache_dir),
        "--detection-json",
        str(detection_a),
        "--detection-json",
        str(detection_b),
        "--nms-translation-threshold-mm",
        "5",
        "--max-images",
        "1",
        "--work-dir",
        str(work_dir),
        "--output",
        str(output),
    ])

    run_bop.main()
    report = json.loads(capsys.readouterr().out)

    assert report["processed_image_count"] == 1
    assert report["processed_target_count"] == 2
    assert report["prediction_count"] == 2
    assert report["output"] == str(output)
    assert len(calls) == 2
    assert [int(call[call.index("--obj-id") + 1]) for call in calls] == [3, 4]

    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(int(row["im_id"]), int(row["obj_id"])) for row in rows] == [
        (2, 3),
        (2, 4),
    ]
    assert [float(row["score"]) for row in rows] == [1.0, 0.9]
    assert [float(x) for x in rows[0]["R"].split()] == np.eye(3).reshape(-1).tolist()
    assert [float(x) for x in rows[0]["t"].split()] == [3.0, 20.0, 30.0]
    assert float(rows[0]["time"]) >= 0.0
    assert rows[0]["time"] == rows[1]["time"]
