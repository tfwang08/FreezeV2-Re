#!/usr/bin/env bash
set -euo pipefail

GEDI_COMMIT="b3dd86776750d8221f89d39975118da9839b39f7"
GEDI_ROOT="${FREEZEV2_GEDI_ROOT:-external/gedi}"

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
import torch

print("[FreezeV2-Re] python:", sys.version.split()[0])
print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] torch CUDA:", torch.version.cuda)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[FreezeV2-Re] GPU:", torch.cuda.get_device_name(0))
print("[FreezeV2-Re] CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
PY

TORCH_MM="$(python - <<'PY'
import torch
v = torch.__version__.split('+', 1)[0].split('.')
print('.'.join(v[:2]))
PY
)"

if [[ "$TORCH_MM" != "2.2" && "${FREEZEV2_OPEN3D_READY:-0}" != "1" ]]; then
    cat >&2 <<EOF
[FreezeV2-Re] STOP before modifying the freeze environment.
The Open3D 0.19 PyPI wheel exposes open3d.ml.torch against PyTorch 2.2.x,
but the current freeze environment has PyTorch $TORCH_MM.

Do not downgrade torch. Build Open3D PyTorch ops against the current freeze
PyTorch first; after that rerun with FREEZEV2_OPEN3D_READY=1.
EOF
    exit 2
fi

if [[ ! -d "$GEDI_ROOT/.git" ]]; then
    git clone https://github.com/fabiopoiesi/gedi.git "$GEDI_ROOT"
fi
git -C "$GEDI_ROOT" fetch --all --tags
git -C "$GEDI_ROOT" checkout "$GEDI_COMMIT"

python -m pip install --upgrade setuptools wheel ninja
python -m pip install torchgeometry==0.1.2 gdown

if [[ "$TORCH_MM" == "2.2" ]]; then
    python -m pip install open3d==0.19.0
fi

python - <<'PY'
import open3d.ml.torch as ml3d
import torch

print("[FreezeV2-Re] Open3D ML import: OK")
print("[FreezeV2-Re] radius_search:", ml3d.ops.radius_search)
print("[FreezeV2-Re] torch unchanged:", torch.__version__)
PY

if command -v nvcc >/dev/null 2>&1; then
    echo "[FreezeV2-Re] nvcc: $(nvcc --version | tail -n 1)"
else
    echo "[FreezeV2-Re] WARNING: nvcc not found; official PointNet2 CUDA build will fail." >&2
fi

python -m pip install "$GEDI_ROOT/backbones/pointnet2_ops_lib/"
python "$GEDI_ROOT/download_data.py"

python - <<'PY'
import torch
from pointnet2_ops.pointnet2_modules import PointnetSAModule

print("[FreezeV2-Re] PointNet2 import: OK")
print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
PY

echo "[FreezeV2-Re] GeDi dependencies installed in the freeze environment."
