from __future__ import annotations

from pathlib import Path

import numpy as np


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
