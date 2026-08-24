from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def _fmt(x: float) -> str:
    return f"{float(x):.9g}"


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
