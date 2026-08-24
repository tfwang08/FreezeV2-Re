from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

CORE_DATASETS = ("lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv")
BOP_TOOLKIT_COMMIT = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"

REFERENCE_SUBMISSIONS = {
    "lmo": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_lmo-test_e2e1e0fc-bec1-46c8-bfac-111bda0ea6ea.csv",
        "filename": "freezev21_lmo-test.csv",
        "expected": {"ar": 0.771, "vsd": 0.623, "mssd": 0.829, "mspd": 0.861, "time": 29.805},
    },
    "tless": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_tless-test_b1f90f3f-1e99-46d5-9464-9eef29f1d412.csv",
        "filename": "freezev21_tless-test.csv",
        "expected": {"ar": 0.755, "vsd": 0.708, "mssd": 0.768, "mspd": 0.788, "time": 20.936},
    },
    "tudl": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_tudl-test_09218151-4f7c-4318-a3f7-916a55e78897.csv",
        "filename": "freezev21_tudl-test.csv",
        "expected": {"ar": 0.976, "vsd": 0.940, "mssd": 0.993, "mspd": 0.996, "time": 4.534},
    },
    "icbin": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_icbin-test_903a2ede-25ea-4afd-9645-5e43617e637a.csv",
        "filename": "freezev21_icbin-test.csv",
        "expected": {"ar": 0.697, "vsd": 0.671, "mssd": 0.714, "mspd": 0.705, "time": 46.732},
    },
    "itodd": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_itodd-test_3d8dcb13-6aa8-46b2-aa7b-e207146b85fe.csv",
        "filename": "freezev21_itodd-test.csv",
        "expected": {"ar": 0.742, "vsd": 0.661, "mssd": 0.787, "mspd": 0.777, "time": 24.422},
    },
    "hb": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_hb-test_2dd5a3a2-1fc9-46a4-9fa9-f56d2da71b95.csv",
        "filename": "freezev21_hb-test.csv",
        "expected": {"ar": 0.892, "vsd": 0.855, "mssd": 0.908, "mspd": 0.912, "time": 21.055},
    },
    "ycbv": {
        "url": "https://bop.felk.cvut.cz/media/subs/freezev21_ycbv-test_d0e9610c-9272-4216-a1f3-69ca81e6c748.csv",
        "filename": "freezev21_ycbv-test.csv",
        "expected": {"ar": 0.915, "vsd": 0.888, "mssd": 0.948, "mspd": 0.908, "time": 26.760},
    },
}


def _fmt(x: float) -> str:
    return f"{float(x):.9g}"


def _check_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset not in CORE_DATASETS:
        raise ValueError(f"Unknown BOP core dataset: {dataset}")
    return dataset


def bop_dataset_patterns(dataset: str) -> list[str]:
    dataset = _check_dataset(dataset)
    return [f"{dataset}_base.zip", f"{dataset}_models.zip", "*test*bop19.zip"]


def load_bop_targets(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        targets = json.load(f)
    if not isinstance(targets, list):
        raise ValueError("BOP targets file must contain a JSON list")
    return targets


def validate_bop_result_rows(rows: list[dict]) -> None:
    required = {"scene_id", "im_id", "obj_id", "score", "R", "t", "time"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"row {index} missing BOP fields: {sorted(missing)}")
        try:
            int(row["scene_id"])
            int(row["im_id"])
            int(row["obj_id"])
            float(row["score"])
            float(row["time"])
            rotation = [float(x) for x in str(row["R"]).split()]
            translation = [float(x) for x in str(row["t"]).split()]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {index} contains non-numeric BOP values") from exc
        if len(rotation) != 9:
            raise ValueError(f"row {index} rotation must contain 9 values")
        if len(translation) != 3:
            raise ValueError(f"row {index} translation must contain 3 values")


def read_bop_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    validate_bop_result_rows(rows)
    return rows


def write_bop_csv(path: str | Path, predictions: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in predictions:
            pose = np.asarray(p["pose"], dtype=np.float64)
            writer.writerow({
                "scene_id": int(p["scene_id"]),
                "im_id": int(p["im_id"]),
                "obj_id": int(p["obj_id"]),
                "score": _fmt(p["score"]),
                "R": " ".join(_fmt(x) for x in pose[:3, :3].reshape(-1)),
                "t": " ".join(_fmt(x) for x in pose[:3, 3]),
                "time": _fmt(p.get("time", -1.0)),
            })


def prepare_bop_dataset(dataset: str, bop_root: str | Path) -> Path:
    dataset = _check_dataset(dataset)
    bop_root = Path(bop_root)
    archive_dir = bop_root / "_archives" / dataset
    archive_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with: pip install -e '.[data]'") from exc

    snapshot_download(
        repo_id=f"bop-benchmark/{dataset}",
        repo_type="dataset",
        local_dir=archive_dir,
        allow_patterns=bop_dataset_patterns(dataset),
    )

    base_zip = archive_dir / f"{dataset}_base.zip"
    models_zip = archive_dir / f"{dataset}_models.zip"
    test_zips = sorted(archive_dir.glob("*test*bop19.zip"))
    if not base_zip.exists() or not models_zip.exists() or len(test_zips) != 1:
        raise RuntimeError(f"Incomplete BOP archives for {dataset} in {archive_dir}")

    bop_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as zf:
        zf.extractall(bop_root)
    dataset_root = bop_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    for archive in (models_zip, test_zips[0]):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dataset_root)

    targets = dataset_root / "test_targets_bop19.json"
    if not targets.exists():
        raise RuntimeError(f"Missing {targets} after extraction")
    return dataset_root


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "FreezeV2-Re/0.1"})
    with urllib.request.urlopen(request) as response, path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def download_reference_submission(dataset: str, output_dir: str | Path) -> Path:
    dataset = _check_dataset(dataset)
    spec = REFERENCE_SUBMISSIONS[dataset]
    output_path = Path(output_dir) / spec["filename"]
    if not output_path.exists():
        _download(spec["url"], output_path)
    read_bop_csv(output_path)
    return output_path


def build_eval_command(
    bop_toolkit: str | Path,
    bop_root: str | Path,
    result_csv: str | Path,
    eval_root: str | Path,
    num_workers: int = 10,
) -> tuple[list[str], dict[str, str]]:
    bop_toolkit = Path(bop_toolkit)
    result_csv = Path(result_csv)
    eval_root = Path(eval_root)
    cmd = [
        sys.executable,
        str(bop_toolkit / "scripts" / "eval_bop19_pose.py"),
        "--renderer_type=vispy",
        f"--result_filenames={result_csv.name}",
        f"--results_path={result_csv.parent}",
        f"--eval_path={eval_root}",
        "--targets_filename=test_targets_bop19.json",
        f"--num_workers={int(num_workers)}",
    ]
    env = os.environ.copy()
    env["BOP_PATH"] = str(Path(bop_root))
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("EGL_PLATFORM", "surfaceless")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    return cmd, env


def evaluate_reference(
    dataset: str,
    bop_root: str | Path,
    result_csv: str | Path,
    bop_toolkit: str | Path,
    eval_root: str | Path,
    num_workers: int = 10,
) -> Path:
    dataset = _check_dataset(dataset)
    result_csv = Path(result_csv)
    expected_name = REFERENCE_SUBMISSIONS[dataset]["filename"]
    if result_csv.name != expected_name:
        raise ValueError(f"Reference result must be named {expected_name} for BOP filename parsing")
    read_bop_csv(result_csv)
    cmd, env = build_eval_command(bop_toolkit, bop_root, result_csv, eval_root, num_workers)
    subprocess.run(cmd, check=True, env=env)
    score_path = Path(eval_root) / result_csv.stem / "scores_bop19.json"
    if not score_path.exists():
        raise RuntimeError(f"Official BOP evaluator did not create {score_path}")
    return score_path
