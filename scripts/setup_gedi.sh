#!/usr/bin/env bash
set -euo pipefail

GEDI_COMMIT="b3dd86776750d8221f89d39975118da9839b39f7"
OPEN3D_COMMIT="22a6a307b9b7d88604895f79c0ceeecef2fc6538"
GEDI_ROOT="${FREEZEV2_GEDI_ROOT:-external/gedi}"
OPEN3D_ROOT="${FREEZEV2_OPEN3D_ROOT:-external/open3d}"
OPEN3D_BUILD="${FREEZEV2_OPEN3D_BUILD:-$OPEN3D_ROOT/build-freeze}"
BUILD_JOBS="${FREEZEV2_BUILD_JOBS:-8}"

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
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for GeDi")
PY

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[FreezeV2-Re] nvcc is required to build Open3D and PointNet2 CUDA ops." >&2
    exit 2
fi
echo "[FreezeV2-Re] nvcc: $(nvcc --version | tail -n 1)"

TORCH_ABI="$(python - <<'PY'
import torch
print("ON" if torch._C._GLIBCXX_USE_CXX11_ABI else "OFF")
PY
)"

# Build Open3D's official PyTorch ops against the PyTorch already installed in
# freeze. Do not install the released Open3D wheel because its torch ABI/version
# is unrelated to this environment.
python -m pip install "cmake>=3.24" ninja wheel setuptools

if [[ ! -d "$OPEN3D_ROOT/.git" ]]; then
    git clone https://github.com/isl-org/Open3D.git "$OPEN3D_ROOT"
fi
git -C "$OPEN3D_ROOT" fetch --all --tags
git -C "$OPEN3D_ROOT" checkout "$OPEN3D_COMMIT"

cmake -S "$OPEN3D_ROOT" -B "$OPEN3D_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PYTHON_MODULE=ON \
    -DBUILD_CUDA_MODULE=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_OPEN3D_ML=OFF \
    -DBUILD_GUI=OFF \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_LIBREALSENSE=OFF \
    -DBUILD_AZURE_KINECT=OFF \
    -DWITH_STUBGEN=OFF \
    -DDEVELOPER_BUILD=ON \
    -DGLIBCXX_USE_CXX11_ABI="$TORCH_ABI" \
    -DPython3_EXECUTABLE="$(command -v python)"

cmake --build "$OPEN3D_BUILD" --target pip-package --parallel "$BUILD_JOBS"

OPEN3D_WHEEL="$(find "$OPEN3D_BUILD/lib/python_package/pip_package" -maxdepth 1 -name 'open3d*.whl' -print | head -n 1)"
if [[ -z "$OPEN3D_WHEEL" || ! -f "$OPEN3D_WHEEL" ]]; then
    echo "[FreezeV2-Re] Open3D wheel was not produced." >&2
    exit 3
fi

# --no-deps is intentional: this build must not replace torch or other freeze
# packages. The wheel contains the Open3D core and official torch ops.
python -m pip install --force-reinstall --no-deps "$OPEN3D_WHEEL"

python - <<'PY'
import open3d
import open3d.ml.torch as ml3d
import torch

print("[FreezeV2-Re] Open3D:", open3d.__version__)
print("[FreezeV2-Re] Open3D CUDA:", open3d._build_config["CUDA_VERSION"])
print("[FreezeV2-Re] Open3D PyTorch:", open3d._build_config["Pytorch_VERSION"])
print("[FreezeV2-Re] Open3D CXX11 ABI:", open3d.pybind._GLIBCXX_USE_CXX11_ABI)
print("[FreezeV2-Re] Open3D ML loaded:", ml3d._loaded)
print("[FreezeV2-Re] radius_search:", ml3d.ops.radius_search)
print("[FreezeV2-Re] torch unchanged:", torch.__version__)
assert ml3d._loaded
assert open3d.pybind._GLIBCXX_USE_CXX11_ABI == torch._C._GLIBCXX_USE_CXX11_ABI
PY

if [[ ! -d "$GEDI_ROOT/.git" ]]; then
    git clone https://github.com/fabiopoiesi/gedi.git "$GEDI_ROOT"
fi
git -C "$GEDI_ROOT" fetch --all --tags
git -C "$GEDI_ROOT" checkout "$GEDI_COMMIT"

python -m pip install torchgeometry==0.1.2 gdown

# GeDi vendors the official PointNet2 extension. Its 2022 setup.py hard-codes
# CUDA architectures 3.7..7.5, which CUDA 13 can no longer compile. This patch
# changes build metadata only: the PointNet2 Python API and CUDA/C++ kernels are
# untouched. A800 is compute capability 8.0.
python - "$GEDI_ROOT/backbones/pointnet2_ops_lib/setup.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
new = 'os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0")'
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit("unexpected PointNet2 setup.py; refusing to patch")
PY

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
python -m pip install --no-build-isolation "$GEDI_ROOT/backbones/pointnet2_ops_lib/"
python "$GEDI_ROOT/download_data.py"

python - <<'PY'
import open3d.ml.torch as ml3d
import torch
import torchgeometry
from pointnet2_ops.pointnet2_modules import PointnetSAModule

print("[FreezeV2-Re] GeDi runtime imports: OK")
print("[FreezeV2-Re] PointNet2:", PointnetSAModule)
print("[FreezeV2-Re] radius_search:", ml3d.ops.radius_search)
print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
PY

echo "[FreezeV2-Re] GeDi dependencies installed in the freeze environment."
