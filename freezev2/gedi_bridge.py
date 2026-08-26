from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .gedi_radius import ml3d


GEDI_REPO_COMMIT = "b3dd86776750d8221f89d39975118da9839b39f7"
GEDI_SCALES = (0.30, 0.40)


def build_gedi_config(
    object_diameter: float,
    scale: float,
    checkpoint: str | Path,
) -> dict:
    """Build the official GeDi inference config at one FreeZe neighbourhood scale."""
    object_diameter = float(object_diameter)
    scale = float(scale)
    if object_diameter <= 0:
        raise ValueError("object_diameter must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")

    return {
        "dim": 32,
        "samples_per_batch": 500,
        "samples_per_patch_lrf": 4000,
        "samples_per_patch_out": 512,
        "r_lrf": scale * object_diameter,
        "fchkpt_gedi_net": str(checkpoint),
    }


def concatenate_gedi_scales(features_30, features_40) -> np.ndarray:
    """Concatenate the two 32D FreeZe GeDi scales into one 64D branch."""
    features_30 = np.asarray(features_30, dtype=np.float32)
    features_40 = np.asarray(features_40, dtype=np.float32)
    if features_30.ndim != 2 or features_40.ndim != 2:
        raise ValueError("GeDi features must have shape Nx32")
    if features_30.shape != features_40.shape:
        raise ValueError("30% and 40% GeDi features must have the same shape")
    if features_30.shape[1] != 32:
        raise ValueError("each GeDi scale must be 32-dimensional")
    if not np.isfinite(features_30).all() or not np.isfinite(features_40).all():
        raise ValueError("GeDi features must be finite")
    return np.concatenate([features_30, features_40], axis=1).astype(np.float32)


def _as_points(points, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


@contextmanager
def _radius_search_import_shim():
    """Temporarily satisfy GeDi's Open3D-ML import with our ABI-free ops shim."""
    names = ("open3d", "open3d.ml", "open3d.ml.torch")
    previous = {name: sys.modules.get(name) for name in names}

    open3d = types.ModuleType("open3d")
    open3d.__path__ = []
    open3d_ml = types.ModuleType("open3d.ml")
    open3d_ml.__path__ = []
    open3d_ml_torch = types.ModuleType("open3d.ml.torch")
    open3d_ml_torch.ops = ml3d.ops
    open3d.ml = open3d_ml
    open3d_ml.torch = open3d_ml_torch

    sys.modules["open3d"] = open3d
    sys.modules["open3d.ml"] = open3d_ml
    sys.modules["open3d.ml.torch"] = open3d_ml_torch
    try:
        yield
    finally:
        for name in names:
            old = previous[name]
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_official_gedi_class(gedi_root: str | Path):
    """Load pinned official GeDi code while replacing only radius_search."""
    gedi_root = Path(gedi_root).resolve()
    gedi_path = gedi_root / "gedi.py"
    if not gedi_path.is_file():
        raise FileNotFoundError(gedi_path)

    module_name = "_freezev2_official_gedi"
    spec = importlib.util.spec_from_file_location(module_name, gedi_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official GeDi module from {gedi_path}")
    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, str(gedi_root))
    try:
        with _radius_search_import_shim():
            spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(gedi_root))
        except ValueError:
            pass
    return module.GeDi


class GediExtractor:
    """Frozen adapter around the pinned official GeDi implementation.

    The official GeDi implementation and PointNet2 network remain unchanged.
    Only its Open3D-ML radius-search dependency is replaced by an equivalent
    CPU radius search so the extractor is not tied to Open3D's PyTorch ABI.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        gedi_root: str | Path = "external/gedi",
        device: str = "cuda",
        seed: int = 0,
        backend=None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.gedi_root = Path(gedi_root)
        self.device = str(device)
        self.seed = int(seed)
        if not self.device.startswith("cuda"):
            raise ValueError("official GeDi inference requires a CUDA device")

        if backend is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(self.checkpoint)
            if not self.gedi_root.is_dir():
                raise FileNotFoundError(self.gedi_root)

            head = subprocess.check_output(
                ["git", "-C", str(self.gedi_root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            if head != GEDI_REPO_COMMIT:
                raise RuntimeError(
                    "external GeDi revision mismatch: "
                    f"expected {GEDI_REPO_COMMIT}, got {head}"
                )

            import torch

            backend = (torch, _load_official_gedi_class(self.gedi_root))

        self._torch, self._gedi_cls = backend
        if not self._torch.cuda.is_available():
            raise RuntimeError("official GeDi inference requires CUDA")

    def encode(
        self,
        points,
        cloud,
        object_diameter: float,
    ) -> np.ndarray:
        """Return FreeZeV2 two-scale GeDi descriptors with shape Nx64."""
        points = _as_points(points, "points")
        cloud = _as_points(cloud, "cloud")
        object_diameter = float(object_diameter)
        if object_diameter <= 0:
            raise ValueError("object_diameter must be positive")

        np.random.seed(self.seed)
        self._torch.manual_seed(self.seed)
        self._torch.cuda.manual_seed_all(self.seed)

        points_t = self._torch.from_numpy(points).float()
        cloud_t = self._torch.from_numpy(cloud).float()
        per_scale = []

        for scale in GEDI_SCALES:
            model = self._gedi_cls(
                config=build_gedi_config(
                    object_diameter,
                    scale,
                    self.checkpoint,
                )
            )
            descriptors = np.asarray(
                model.compute(pts=points_t, pcd=cloud_t),
                dtype=np.float32,
            )
            if descriptors.shape != (len(points), 32):
                raise RuntimeError(
                    "unexpected GeDi output shape at scale "
                    f"{scale:.2f}: {descriptors.shape}"
                )
            if not np.isfinite(descriptors).all():
                raise RuntimeError(
                    f"GeDi output contains non-finite values at scale {scale:.2f}"
                )
            per_scale.append(descriptors)
            del model
            self._torch.cuda.empty_cache()

        return concatenate_gedi_scales(per_scale[0], per_scale[1])
