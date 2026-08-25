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

if ! conda env list | awk '{print $1}' | grep -qx gedi; then
    conda create -y -n gedi python=3.8
fi
conda activate gedi

python -m pip install --upgrade "pip<24.1" setuptools wheel ninja
python -m pip install \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install \
    "numpy<2" \
    open3d==0.15.1 \
    torchgeometry==0.1.2 \
    gdown \
    "protobuf==3.20.*"

if [[ ! -d "$GEDI_ROOT/.git" ]]; then
    git clone https://github.com/fabiopoiesi/gedi.git "$GEDI_ROOT"
fi
git -C "$GEDI_ROOT" fetch --all --tags
git -C "$GEDI_ROOT" checkout "$GEDI_COMMIT"

if command -v nvcc >/dev/null 2>&1; then
    echo "[FreezeV2-Re] nvcc: $(nvcc --version | tail -n 1)"
else
    echo "[FreezeV2-Re] WARNING: nvcc not found; PointNet2 CUDA build will fail." >&2
fi

python -m pip install "$GEDI_ROOT/backbones/pointnet2_ops_lib/"
python "$GEDI_ROOT/download_data.py"

python - <<'PY'
import torch
print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[FreezeV2-Re] GPU:", torch.cuda.get_device_name(0))
PY

echo "[FreezeV2-Re] GeDi environment ready."
echo "[FreezeV2-Re] Enter it with: source scripts/use_gedi.sh"
