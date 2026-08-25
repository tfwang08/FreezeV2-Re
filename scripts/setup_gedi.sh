#!/usr/bin/env bash
set -eo pipefail

# Prebuilt-only GeDi environment setup.
#
# Do not compile Open3D or PointNet2 in this script. Open3D 0.19.0's released
# wheel was built for PyTorch 2.2.*, so freeze is pinned to that binary stack.
# The official GeDi checkpoint is also treated as an explicit local input:
# cluster nodes may not have network access to Google Drive.

GEDI_COMMIT="b3dd86776750d8221f89d39975118da9839b39f7"
GEDI_ROOT="${FREEZEV2_GEDI_ROOT:-external/gedi}"

TORCH_VERSION="${FREEZEV2_TORCH_VERSION:-2.2.2}"
TORCHVISION_VERSION="${FREEZEV2_TORCHVISION_VERSION:-0.17.2}"
TORCHAUDIO_VERSION="${FREEZEV2_TORCHAUDIO_VERSION:-2.2.2}"
NUMPY_VERSION="${FREEZEV2_NUMPY_VERSION:-1.26.4}"
OPEN3D_VERSION="${FREEZEV2_OPEN3D_VERSION:-0.19.0}"

PIP_INDEX_URL="${FREEZEV2_PIP_INDEX_URL:-https://mirrors.cloud.aliyuncs.com/pypi/simple}"
PIP_TRUSTED_HOST="${FREEZEV2_PIP_TRUSTED_HOST:-mirrors.cloud.aliyuncs.com}"
PYTORCH_INDEX_URL="${FREEZEV2_PYTORCH_INDEX_URL:-https://mirrors.cloud.aliyuncs.com/pytorch-wheels/cu121}"

_pip_clean() {
    env \
        -u PIP_INDEX_URL \
        -u PIP_EXTRA_INDEX_URL \
        -u PIP_TRUSTED_HOST \
        -u PIP_NO_INDEX \
        -u PIP_FIND_LINKS \
        PIP_CONFIG_FILE=/dev/null \
        python -m pip "$@"
}

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
echo "[FreezeV2-Re] PyPI index: $PIP_INDEX_URL"

# PyTorch's CUDA 12.1 wheel reports 2.2.2+cu121 after installation.
_pip_clean install \
    --force-reinstall \
    --no-cache-dir \
    --index-url "$PYTORCH_INDEX_URL" \
    --extra-index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

_pip_clean install \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    --upgrade \
    "numpy==$NUMPY_VERSION" \
    "open3d==$OPEN3D_VERSION" \
    "torchgeometry==0.1.2"

python - <<'PY'
import numpy as np
import open3d
import open3d.ml.torch as ml3d
import torch

print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] torch CUDA:", torch.version.cuda)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[FreezeV2-Re] GPU:", torch.cuda.get_device_name(0))
print("[FreezeV2-Re] NumPy:", np.__version__)
print("[FreezeV2-Re] Open3D:", open3d.__version__)
print("[FreezeV2-Re] Open3D CUDA available:", open3d.core.cuda.is_available())
print("[FreezeV2-Re] Open3D ML loaded:", ml3d._loaded)
print("[FreezeV2-Re] radius_search:", ml3d.ops.radius_search)

if torch.__version__ != "2.2.2+cu121":
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda != "12.1" or not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA 12.1 is unavailable")
if np.__version__ != "1.26.4":
    raise SystemExit(f"unexpected NumPy version: {np.__version__}")
if open3d.__version__ != "0.19.0":
    raise SystemExit(f"unexpected Open3D version: {open3d.__version__}")
if not ml3d._loaded or not open3d.core.cuda.is_available():
    raise SystemExit("released Open3D CUDA/PyTorch ops are unavailable")
PY

if [[ ! -d "$GEDI_ROOT/.git" ]]; then
    git clone https://github.com/fabiopoiesi/gedi.git "$GEDI_ROOT"
fi
git -C "$GEDI_ROOT" fetch --all --tags
git -C "$GEDI_ROOT" checkout "$GEDI_COMMIT"

# FreeZeV2 only needs the GeDi 3DMatch checkpoint. The assets downloaded by
# GeDi's demo script are not used by this reproduction.
GEDI_CHECKPOINT_CANONICAL="$GEDI_ROOT/data/chkpts/3dmatch/chkpt.tar"
GEDI_CHECKPOINT_INPUT="${FREEZEV2_GEDI_CHECKPOINT:-$GEDI_CHECKPOINT_CANONICAL}"

if [[ "$GEDI_CHECKPOINT_INPUT" != "$GEDI_CHECKPOINT_CANONICAL" ]]; then
    if [[ ! -s "$GEDI_CHECKPOINT_INPUT" ]]; then
        echo "[FreezeV2-Re] GeDi checkpoint not found: $GEDI_CHECKPOINT_INPUT" >&2
        exit 5
    fi
    mkdir -p "$(dirname "$GEDI_CHECKPOINT_CANONICAL")"
    GEDI_CHECKPOINT_ABS="$(python - "$GEDI_CHECKPOINT_INPUT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
    ln -sfn "$GEDI_CHECKPOINT_ABS" "$GEDI_CHECKPOINT_CANONICAL"
fi

if [[ ! -s "$GEDI_CHECKPOINT_CANONICAL" ]]; then
    cat >&2 <<EOF
[FreezeV2-Re] STOP: the official GeDi 3DMatch checkpoint is not present.
[FreezeV2-Re] Required file: $GEDI_CHECKPOINT_CANONICAL
[FreezeV2-Re] The cluster cannot reach drive.google.com, so automatic gdown
[FreezeV2-Re] download is intentionally disabled. Download the official GeDi
[FreezeV2-Re] checkpoint bundle on a machine with Google Drive access, extract
[FreezeV2-Re] chkpts/3dmatch/chkpt.tar, then either place it at the path above
[FreezeV2-Re] or set FREEZEV2_GEDI_CHECKPOINT=/path/to/chkpt.tar.
[FreezeV2-Re] Official Google Drive file id: 1Lpep5QigALjk60h8bNJAUH3DnxtnGcZX
EOF
    exit 5
fi

echo "[FreezeV2-Re] GeDi checkpoint: $GEDI_CHECKPOINT_CANONICAL"

POINTNET2_WHEEL="${FREEZEV2_POINTNET2_WHEEL:-}"
if [[ -z "$POINTNET2_WHEEL" ]]; then
    cat >&2 <<'EOF'
[FreezeV2-Re] STOP: Open3D and the GeDi checkpoint are ready, but GeDi also
[FreezeV2-Re] needs the PointNet2 CUDA extension. Source/JIT compilation is
[FreezeV2-Re] intentionally disabled. Set FREEZEV2_POINTNET2_WHEEL only after
[FreezeV2-Re] an exact Python 3.12 / PyTorch 2.2 / CUDA 12.1 wheel is identified.
EOF
    exit 4
fi

_pip_clean install --no-deps "$POINTNET2_WHEEL"

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
