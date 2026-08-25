from pathlib import Path

import numpy as np

from freezev2.gedi_bridge import (
    GEDI_REPO_COMMIT,
    build_gedi_config,
    concatenate_gedi_scales,
)


def test_gedi_config_uses_freeze_scales_and_official_inference_shape():
    cfg30 = build_gedi_config(100.0, 0.30, Path("checkpoint.tar"))
    cfg40 = build_gedi_config(100.0, 0.40, Path("checkpoint.tar"))

    assert cfg30["dim"] == 32
    assert cfg30["samples_per_batch"] == 500
    assert cfg30["samples_per_patch_lrf"] == 4000
    assert cfg30["samples_per_patch_out"] == 512
    assert cfg30["r_lrf"] == 30.0
    assert cfg40["r_lrf"] == 40.0


def test_gedi_scale_concatenation_is_64d_and_ordered_30_then_40():
    f30 = np.tile(np.arange(32), (3, 1)).astype(np.float32)
    f40 = np.tile(np.arange(32, 64), (3, 1)).astype(np.float32)

    fused = concatenate_gedi_scales(f30, f40)

    assert fused.shape == (3, 64)
    np.testing.assert_array_equal(fused[:, :32], f30)
    np.testing.assert_array_equal(fused[:, 32:], f40)


def test_official_gedi_revision_is_pinned():
    assert GEDI_REPO_COMMIT == "b3dd86776750d8221f89d39975118da9839b39f7"
