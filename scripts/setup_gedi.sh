#!/usr/bin/env bash
set -eo pipefail

# Prebuilt-only GeDi environment setup.
#
# Do not compile Open3D or PointNet2 in this script. Open3D 0.19.0's released
# wheel was built for PyTorch 2.2.*, so the existing freeze environment is
# intentionally pinned to that compatible binary stack.

GEDI_COMMIT="b3dd86776750d8221f89d39975118da9839b39f7"
GEDI_ROOT="${FREEZEV2_GEDI_ROOT:-external/gedi}"

TORCH_VERSION="${FREEZEV2_TORCH_VERSION:-2.2.2+cu121}"
TORCHVISION_VERSION="${FREEZEV2_TORCHVISION_VERSION:-0.17.2+cu121}"
TORCHAUDIO_VERSION="${FREEZEV2_TORCHAUDIO_VERSION:-2.2.2+cu121}"
NUMPY_VERSION="${FREEZEV2_NUMPY_VERSION:-1.26.4}"
OPEN3D_VERSION="${FREEZEV2_OPEN3D_VERSION:-0.19.0}"

PIP_INDEX_URL="${FREEZEV2_PIP_INDEX_URL:-https://mirrors.cloud.aliyuncs.com/pypi/simple}"
PIP_TRUSTED_HOST="${FREEZEV2_PIP_TRUSTED_HOST:-mirrors.cloud.aliyuncs.com}"
PYTORCH_INDEX_URL="${FREEZEV2_PYTORCH_INDEX_URL:-https://mirrors.cloud.aliyuncs.com/pytorch-wheels/cu121}"
PIP_ARGS=(-i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST")

_load_conda() {
    if declare -F conda >/dev/null 2>&1; then
        return 0
    fi

    local conda_sh=""
    if [[ -n "${CONDA_EXE:-}" ]]; then
        conda_sh="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    fi

    for candidate in \
        "$conda_sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "/workspace/$USER/miniconda3/etc/profile.d/conda.sh"
    do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            # shellcheck disable=SC1090
            source "$candidate"
            return 0
        fi
    done

    echo "Could not find conda.sh. Initialize conda first." >&2
    return 1
}

_load_conda
conda activate freeze

python - <<'PY'
import sys
try:
    import torch
    print("[FreezeV2-Re] current torch:", torch.__version__)
    print("[FreezeV2-Re] current torch CUDA:", torch.version.cuda)
except Exception as exc:
    print("[FreezeV2-Re] current torch import failed:", exc)
print("[FreezeV2-Re] python:", sys.version.split()[0])
PY

echo "[FreezeV2-Re] switching freeze to the released Open3D-compatible binary stack."
echo "[FreezeV2-Re] PyTorch index: $PYTORCH_INDEX_URL"

# PyTorch 2.2.2 provides a CPython 3.12 CUDA 12.1 wheel. Use the binary wheel
# index directly; do not keep the CUDA 13 PyTorch build because Open3D 0.19.0's
# released ML ops reject PyTorch versions outside 2.2.*.
python -m pip install \
    --force-reinstall \
    --no-cache-dir \
    --index-url "$PYTORCH_INDEX_URL" \
    --extra-index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

# Open3D 0.19.0 was built against NumPy 1.x. Pin NumPy before installing the
# released wheel so pip does not leave a NumPy 2.x runtime in the environment.
python -m pip install "${PIP_ARGS[@]}" \
    --upgrade \
    "numpy==$NUMPY_VERSION" \
    "open3d==$OPEN3D_VERSION" \
    "torchgeometry==0.1.2" \
    gdown

python - <<'PY'
import numpy as np
import open3d
import open3d.ml.torch as ml3d
import torch

expected_torch = "2.2.2+cu121"
expected_numpy = "1.26.4"
expected_open3d = "0.19.0"

print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] torch CUDA:", torch.version.cuda)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[FreezeV2-Re] GPU:", torch.cuda.get_device_name(0))
print("[FreezeV2-Re] NumPy:", np.__version__)
print("[FreezeV2-Re] Open3D:", open3d.__version__)
print("[FreezeV2-Re] Open3D build config:", open3d._build_config)
print("[FreezeV2-Re] Open3D CUDA available:", open3d.core.cuda.is_available())
print("[FreezeV2-Re] Open3D ML loaded:", ml3d._loaded)
print("[FreezeV2-Re] radius_search:", ml3d.ops.radius_search)

if torch.__version__ != expected_torch:
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda != "12.1":
    raise SystemExit(f"unexpected torch CUDA version: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
if np.__version__ != expected_numpy:
    raise SystemExit(f"unexpected NumPy version: {np.__version__}")
if open3d.__version__ != expected_open3d:
    raise SystemExit(f"unexpected Open3D version: {open3d.__version__}")
if not ml3d._loaded:
    raise SystemExit("Open3D PyTorch ops failed to load")
if not open3d.core.cuda.is_available():
    raise SystemExit("released Open3D wheel has no usable CUDA backend")
PY

# Keep using the official GeDi source/checkpoint, but do not build its vendored
# PointNet2 extension. A prebuilt PointNet2 wheel must be supplied explicitly.
if [[ ! -d "$GEDI_ROOT/.git" ]]; then
    git clone https://github.com/fabiopoiesi/gedi.git "$GEDI_ROOT"
fi
git -C "$GEDI_ROOT" fetch --all --tags
git -C "$GEDI_ROOT" checkout "$GEDI_COMMIT"
python "$GEDI_ROOT/download_data.py"

POINTNET2_WHEEL="${FREEZEV2_POINTNET2_WHEEL:-}"
if [[ -z "$POINTNET2_WHEEL" ]]; then
    cat >&2 <<'EOF'
[FreezeV2-Re] STOP: Open3D prebuilt wheel stack is ready, but GeDi also needs
[FreezeV2-Re] the PointNet2 CUDA extension. Official GeDi does not ship a
[FreezeV2-Re] prebuilt wheel for Python 3.12 / PyTorch 2.2 / CUDA 12.1.
[FreezeV2-Re] Source/JIT compilation is intentionally disabled.
[FreezeV2-Re] Set FREEZEV2_POINTNET2_WHEEL only after an exact compatible
[FreezeV2-Re] prebuilt wheel has been identified.
EOF
    exit 4
fi

python -m pip install "${PIP_ARGS[@]}" --no-deps "$POINTNET2_WHEEL"

# Verify that the supplied binary actually loads against this exact PyTorch/CUDA
# runtime and can execute one CUDA PointNet2 kernel.
python - <<'PY'
import torch
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from pointnet2_ops.pointnet2_utils import furthest_point_sample

xyz = torch.rand(1, 32, 3, device="cuda", dtype=torch.float32)
idx = furthest_point_sample(xyz, 8)
torch.cuda.synchronize()
print("[FreezeV2-Re] PointNet2:", PointnetSAModule)
print("[FreezeV2-Re] PointNet2 CUDA smoke:", tuple(idx.shape), idx.device)
PY

python - "$GEDI_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

import open3d.ml.torch as ml3d
import torch
import torchgeometry
from gedi import GeDi

print("[FreezeV2-Re] GeDi runtime imports: OK")
print("[FreezeV2-Re] GeDi:", GeDi)
print("[FreezeV2-Re] radius_search:", ml3d.ops.radius_search)
print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
PY

echo "[FreezeV2-Re] GeDi prebuilt-only dependencies are ready in freeze."
