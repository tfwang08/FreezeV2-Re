#!/usr/bin/env bash
set -eo pipefail

# Prebuilt-only GeDi environment setup.
#
# This script intentionally does not compile Open3D or PointNet2. The binary
# stack is pinned to Python 3.11 + PyTorch 2.2.2/CUDA 12.1 because:
#   1. Open3D 0.19.0's released ML wheel is compatible with PyTorch 2.2.*.
#   2. The PointNet2 3.0.0 wheel below was built with Python 3.11,
#      PyTorch 2.2.2, CUDA 12.1, and includes sm_80 for the A800.
#   3. Its PointNet2 CUDA/C++ sources match GeDi's vendored ops.
# The official GeDi checkpoint remains an explicit local input because cluster
# nodes may not have access to Google Drive.

GEDI_COMMIT="b3dd86776750d8221f89d39975118da9839b39f7"
GEDI_ROOT="${FREEZEV2_GEDI_ROOT:-external/gedi}"

PYTHON_MINOR="${FREEZEV2_PYTHON_MINOR:-3.11}"
TORCH_VERSION="${FREEZEV2_TORCH_VERSION:-2.2.2}"
TORCHVISION_VERSION="${FREEZEV2_TORCHVISION_VERSION:-0.17.2}"
TORCHAUDIO_VERSION="${FREEZEV2_TORCHAUDIO_VERSION:-2.2.2}"
NUMPY_VERSION="${FREEZEV2_NUMPY_VERSION:-1.26.4}"
OPEN3D_VERSION="${FREEZEV2_OPEN3D_VERSION:-0.19.0}"
POINTNET2_VERSION="3.0.0"
POINTNET2_WHEEL_DEFAULT="https://github.com/YanWenKun/ComfyUI-3D-Pack-LinuxWheels/releases/download/v2/pointnet2_ops-3.0.0-cp311-cp311-linux_x86_64.whl"
POINTNET2_WHEEL="${FREEZEV2_POINTNET2_WHEEL:-$POINTNET2_WHEEL_DEFAULT}"

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

CURRENT_PYTHON_MINOR="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$CURRENT_PYTHON_MINOR" != "$PYTHON_MINOR" ]]; then
    echo "[FreezeV2-Re] switching freeze Python $CURRENT_PYTHON_MINOR -> $PYTHON_MINOR for the prebuilt PointNet2 wheel."
    conda install -y -n freeze "python=$PYTHON_MINOR" pip
    hash -r
    conda activate freeze
fi

python - "$PYTHON_MINOR" <<'PY'
import sys
expected = sys.argv[1]
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
print("[FreezeV2-Re] python:", sys.version.split()[0])
if actual != expected:
    raise SystemExit(f"expected Python {expected}, got {actual}")
PY

echo "[FreezeV2-Re] PyTorch index: $PYTORCH_INDEX_URL"
echo "[FreezeV2-Re] PyPI index: $PIP_INDEX_URL"

_stack_ready() {
    python - <<'PY'
try:
    import numpy as np
    import open3d
    import open3d.ml.torch as ml3d
    import torch
    import torchvision
    import torchaudio
except Exception:
    raise SystemExit(1)

ok = (
    torch.__version__ == "2.2.2+cu121"
    and torch.version.cuda == "12.1"
    and torch.cuda.is_available()
    and torchvision.__version__ == "0.17.2+cu121"
    and torchaudio.__version__ == "2.2.2+cu121"
    and np.__version__ == "1.26.4"
    and open3d.__version__ == "0.19.0"
    and ml3d._loaded
    and open3d.core.cuda.is_available()
)
raise SystemExit(0 if ok else 1)
PY
}

if ! _stack_ready; then
    echo "[FreezeV2-Re] installing released Open3D-compatible binary stack."
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
else
    echo "[FreezeV2-Re] released torch/Open3D binary stack already ready."
fi

python - <<'PY'
import numpy as np
import open3d
import open3d.ml.torch as ml3d
import torch
import torchvision
import torchaudio

print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] torch CUDA:", torch.version.cuda)
print("[FreezeV2-Re] torchvision:", torchvision.__version__)
print("[FreezeV2-Re] torchaudio:", torchaudio.__version__)
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
if torchvision.__version__ != "0.17.2+cu121":
    raise SystemExit(f"unexpected torchvision version: {torchvision.__version__}")
if torchaudio.__version__ != "2.2.2+cu121":
    raise SystemExit(f"unexpected torchaudio version: {torchaudio.__version__}")
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
    cat >&2 <<EOF2
[FreezeV2-Re] STOP: the official GeDi 3DMatch checkpoint is not present.
[FreezeV2-Re] Required file: $GEDI_CHECKPOINT_CANONICAL
[FreezeV2-Re] Place chkpts/3dmatch/chkpt.tar there, or set
[FreezeV2-Re] FREEZEV2_GEDI_CHECKPOINT=/path/to/chkpt.tar.
EOF2
    exit 5
fi

python - "$GEDI_CHECKPOINT_CANONICAL" <<'PY'
import sys
import torch
p = sys.argv[1]
ckpt = torch.load(p, map_location="cpu")
if not isinstance(ckpt, dict) or "pnet_model_state_dict" not in ckpt:
    raise SystemExit("invalid GeDi checkpoint: missing pnet_model_state_dict")
print("[FreezeV2-Re] GeDi checkpoint:", p)
PY

_pointnet_ready() {
    python - <<'PY'
try:
    import torch
    import pointnet2_ops._ext
    from pointnet2_ops._version import __version__
    from pointnet2_ops.pointnet2_utils import furthest_point_sample
    if __version__ != "3.0.0":
        raise RuntimeError(__version__)
    xyz = torch.rand(1, 32, 3, device="cuda", dtype=torch.float32)
    idx = furthest_point_sample(xyz, 8)
    torch.cuda.synchronize()
    if tuple(idx.shape) != (1, 8) or not idx.is_cuda:
        raise RuntimeError("bad PointNet2 CUDA output")
except Exception:
    raise SystemExit(1)
PY
}

if ! _pointnet_ready; then
    echo "[FreezeV2-Re] installing prebuilt PointNet2 $POINTNET2_VERSION wheel."
    echo "[FreezeV2-Re] PointNet2 wheel: $POINTNET2_WHEEL"
    _pip_clean install --force-reinstall --no-deps "$POINTNET2_WHEEL"
fi

python - <<'PY'
import torch
import pointnet2_ops._ext
from pointnet2_ops._version import __version__
from pointnet2_ops.pointnet2_utils import furthest_point_sample

xyz = torch.rand(1, 32, 3, device="cuda", dtype=torch.float32)
idx = furthest_point_sample(xyz, 8)
torch.cuda.synchronize()
print("[FreezeV2-Re] PointNet2 version:", __version__)
print("[FreezeV2-Re] PointNet2 extension:", pointnet2_ops._ext.__file__)
print("[FreezeV2-Re] PointNet2 CUDA smoke:", tuple(idx.shape), idx.device)
if __version__ != "3.0.0":
    raise SystemExit(f"unexpected PointNet2 version: {__version__}")
PY

# End-to-end smoke: use official GeDi Python/network code and checkpoint while
# the installed wheel supplies only the matching prebuilt PointNet2 backend.
python - "$GEDI_ROOT" "$GEDI_CHECKPOINT_CANONICAL" <<'PY'
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1]).resolve()
checkpoint = str(Path(sys.argv[2]).resolve())
sys.path.insert(0, str(root))

import open3d.ml.torch as ml3d
from gedi import GeDi

np.random.seed(0)
torch.manual_seed(0)

config = {
    "dim": 32,
    "samples_per_batch": 2,
    "samples_per_patch_lrf": 512,
    "samples_per_patch_out": 512,
    "r_lrf": 0.5,
    "fchkpt_gedi_net": checkpoint,
}

pcd = torch.rand(600, 3, dtype=torch.float32) * 0.1
pts = pcd[:2].clone()
gedi = GeDi(config=config)
desc = gedi.compute(pts=pts, pcd=pcd)

print("[FreezeV2-Re] GeDi radius_search:", ml3d.ops.radius_search)
print("[FreezeV2-Re] GeDi descriptor smoke:", desc.shape, desc.dtype)
print("[FreezeV2-Re] GeDi descriptor finite:", bool(np.isfinite(desc).all()))

if desc.shape != (2, 32) or not np.isfinite(desc).all():
    raise SystemExit("GeDi end-to-end descriptor smoke failed")
PY

echo "[FreezeV2-Re] GeDi prebuilt-only dependencies are ready in freeze."
