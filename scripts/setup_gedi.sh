#!/usr/bin/env bash
set -eo pipefail

# GeDi environment setup for the supported NVIDIA profiles.
#
# Ampere (A800/A100, sm_80): Python 3.11 + PyTorch 2.2.2/CUDA 12.1.
# Blackwell (RTX 50-series, sm_120): Python 3.11 + PyTorch 2.7.1/CUDA 12.8.
#
# PointNet2 is built from the exact source vendored by the pinned GeDi commit
# for the detected GPU architecture. The matching CUDA compiler, development
# libraries, and a CUDA-compatible GCC/G++ 12 toolchain are installed into an
# isolated prefix instead of the freeze conda environment. This keeps unrelated
# cuda-toolkit/compiler packages in freeze from perturbing the extension build.
# GeDi's Open3D-ML radius search is replaced in freezev2.gedi_bridge by an
# equivalent CPU torch search, so the Blackwell profile is not blocked by
# Open3D 0.19's PyTorch-2.2 ABI.

GEDI_COMMIT="b3dd86776750d8221f89d39975118da9839b39f7"
GEDI_ROOT="${FREEZEV2_GEDI_ROOT:-external/gedi}"
NUMPY_VERSION="${FREEZEV2_NUMPY_VERSION:-1.26.4}"
POINTNET2_VERSION="3.0.0"
PIP_INDEX_URL="${FREEZEV2_PIP_INDEX_URL:-https://mirrors.cloud.aliyuncs.com/pypi/simple}"
PIP_TRUSTED_HOST="${FREEZEV2_PIP_TRUSTED_HOST:-mirrors.cloud.aliyuncs.com}"

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

_detect_compute_capability() {
    if [[ -n "${FREEZEV2_GPU_CC:-}" ]]; then
        printf '%s\n' "$FREEZEV2_GPU_CC"
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local caps
        caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | sort -u || true)"
        if [[ -n "$caps" ]]; then
            if [[ "$(printf '%s\n' "$caps" | wc -l)" -ne 1 ]]; then
                echo "[FreezeV2-Re] mixed GPU compute capabilities detected:" >&2
                printf '%s\n' "$caps" >&2
                echo "[FreezeV2-Re] set FREEZEV2_GPU_CC explicitly for the GPU used by this environment." >&2
                return 2
            fi
            printf '%s\n' "$caps"
            return 0
        fi
    fi

    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable and nvidia-smi did not report compute capability")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
}

_load_conda
conda activate freeze

GPU_CC="$(_detect_compute_capability)"
read -r PROFILE PYTHON_MINOR TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION CUDA_TAG CUDA_VERSION CUDA_CONDA_LABEL POINTNET_ARCH < <(
    python - "$GPU_CC" <<'PY'
import sys
from freezev2.gpu_stack import resolve_gpu_profile

major, minor = map(int, sys.argv[1].split("."))
p = resolve_gpu_profile((major, minor))
print(
    p.name,
    p.python_minor,
    p.torch_version,
    p.torchvision_version,
    p.torchaudio_version,
    p.cuda_tag,
    p.cuda_version,
    p.cuda_conda_label,
    p.pointnet_arch,
)
PY
)

PYTHON_MINOR="${FREEZEV2_PYTHON_MINOR:-$PYTHON_MINOR}"
TORCH_VERSION="${FREEZEV2_TORCH_VERSION:-$TORCH_VERSION}"
TORCHVISION_VERSION="${FREEZEV2_TORCHVISION_VERSION:-$TORCHVISION_VERSION}"
TORCHAUDIO_VERSION="${FREEZEV2_TORCHAUDIO_VERSION:-$TORCHAUDIO_VERSION}"
CUDA_VERSION="${FREEZEV2_CUDA_VERSION:-$CUDA_VERSION}"
CUDA_CONDA_LABEL="${FREEZEV2_CUDA_CONDA_LABEL:-$CUDA_CONDA_LABEL}"
POINTNET_ARCH="${FREEZEV2_POINTNET_ARCH:-$POINTNET_ARCH}"
CUDA_TOOLKIT_PREFIX="${FREEZEV2_CUDA_TOOLKIT_PREFIX:-$HOME/.cache/freezev2/cuda-toolkits/$CUDA_CONDA_LABEL}"

if [[ -n "${FREEZEV2_PYTORCH_INDEX_URL:-}" ]]; then
    PYTORCH_INDEX_URL="$FREEZEV2_PYTORCH_INDEX_URL"
elif [[ "$PROFILE" == "ampere" ]]; then
    PYTORCH_INDEX_URL="https://mirrors.cloud.aliyuncs.com/pytorch-wheels/cu121"
else
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
fi

echo "[FreezeV2-Re] GPU compute capability: $GPU_CC"
echo "[FreezeV2-Re] GeDi GPU profile: $PROFILE"
echo "[FreezeV2-Re] PointNet2 target: sm_${POINTNET_ARCH/./}"
echo "[FreezeV2-Re] CUDA compiler label: $CUDA_CONDA_LABEL"
echo "[FreezeV2-Re] CUDA toolkit prefix: $CUDA_TOOLKIT_PREFIX"
echo "[FreezeV2-Re] PyTorch index: $PYTORCH_INDEX_URL"
echo "[FreezeV2-Re] PyPI index: $PIP_INDEX_URL"

CURRENT_PYTHON_MINOR="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$CURRENT_PYTHON_MINOR" != "$PYTHON_MINOR" ]]; then
    echo "[FreezeV2-Re] switching freeze Python $CURRENT_PYTHON_MINOR -> $PYTHON_MINOR."
    conda install -y -n freeze "python=$PYTHON_MINOR" pip
    hash -r
    conda activate freeze
fi

_stack_ready() {
    python - "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCHAUDIO_VERSION" "$CUDA_TAG" "$CUDA_VERSION" "$GPU_CC" "$NUMPY_VERSION" <<'PY'
import sys

expected_torch, expected_tv, expected_ta, cuda_tag, expected_cuda, gpu_cc, expected_numpy = sys.argv[1:]
try:
    import numpy as np
    import torch
    import torchvision
    import torchaudio
except Exception:
    raise SystemExit(1)

major, minor = map(int, gpu_cc.split("."))
expected_arch = f"sm_{major}{minor}"
ok = (
    torch.__version__ == f"{expected_torch}+{cuda_tag}"
    and torchvision.__version__ == f"{expected_tv}+{cuda_tag}"
    and torchaudio.__version__ == f"{expected_ta}+{cuda_tag}"
    and torch.version.cuda == expected_cuda
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0) == (major, minor)
    and expected_arch in torch.cuda.get_arch_list()
    and np.__version__ == expected_numpy
)
raise SystemExit(0 if ok else 1)
PY
}

if ! _stack_ready; then
    echo "[FreezeV2-Re] installing profile-specific PyTorch stack."
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
        "torchgeometry==0.1.2" \
        "ninja>=1.11" \
        "wheel>=0.43" \
        "setuptools>=68"
else
    echo "[FreezeV2-Re] profile-specific PyTorch stack already ready."
fi

python - "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCHAUDIO_VERSION" "$CUDA_TAG" "$CUDA_VERSION" "$GPU_CC" "$NUMPY_VERSION" <<'PY'
import sys
import numpy as np
import torch
import torchvision
import torchaudio

expected_torch, expected_tv, expected_ta, cuda_tag, expected_cuda, gpu_cc, expected_numpy = sys.argv[1:]
major, minor = map(int, gpu_cc.split("."))
expected_arch = f"sm_{major}{minor}"

print("[FreezeV2-Re] torch:", torch.__version__)
print("[FreezeV2-Re] torch CUDA:", torch.version.cuda)
print("[FreezeV2-Re] torchvision:", torchvision.__version__)
print("[FreezeV2-Re] torchaudio:", torchaudio.__version__)
print("[FreezeV2-Re] NumPy:", np.__version__)
print("[FreezeV2-Re] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[FreezeV2-Re] GPU:", torch.cuda.get_device_name(0))
    print("[FreezeV2-Re] capability:", torch.cuda.get_device_capability(0))
    print("[FreezeV2-Re] torch arch list:", torch.cuda.get_arch_list())

if torch.__version__ != f"{expected_torch}+{cuda_tag}":
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torchvision.__version__ != f"{expected_tv}+{cuda_tag}":
    raise SystemExit(f"unexpected torchvision version: {torchvision.__version__}")
if torchaudio.__version__ != f"{expected_ta}+{cuda_tag}":
    raise SystemExit(f"unexpected torchaudio version: {torchaudio.__version__}")
if torch.version.cuda != expected_cuda or not torch.cuda.is_available():
    raise SystemExit(f"PyTorch CUDA {expected_cuda} is unavailable")
if torch.cuda.get_device_capability(0) != (major, minor):
    raise SystemExit("active GPU does not match the selected profile")
if expected_arch not in torch.cuda.get_arch_list():
    raise SystemExit(f"PyTorch wheel does not contain {expected_arch}")
if np.__version__ != expected_numpy:
    raise SystemExit(f"unexpected NumPy version: {np.__version__}")
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
ckpt = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(ckpt, dict) or "pnet_model_state_dict" not in ckpt:
    raise SystemExit("invalid GeDi checkpoint: missing pnet_model_state_dict")
print("[FreezeV2-Re] GeDi checkpoint:", p)
PY

_nvcc_minor_at() {
    "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1
}

_cuda_header_exists() {
    local root="$1"
    local header="$2"
    [[ -f "$root/include/$header" || -f "$root/targets/x86_64-linux/include/$header" ]]
}

_isolated_toolchain_ready() {
    local nvcc_path="$CUDA_TOOLKIT_PREFIX/bin/nvcc"
    [[ -x "$nvcc_path" ]] || return 1
    [[ "$(_nvcc_minor_at "$nvcc_path")" == "$CUDA_VERSION" ]] || return 1
    _cuda_header_exists "$CUDA_TOOLKIT_PREFIX" "cusparse.h" || return 1
    [[ -x "$CUDA_TOOLKIT_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]] || return 1
    [[ -x "$CUDA_TOOLKIT_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]] || return 1
}

_prepare_cuda_toolkit() {
    local nvcc_path=""
    local current=""

    if [[ -n "${FREEZEV2_CUDA_HOME:-}" ]]; then
        export CUDA_HOME="$FREEZEV2_CUDA_HOME"
        nvcc_path="$CUDA_HOME/bin/nvcc"
        if [[ ! -x "$nvcc_path" ]]; then
            echo "[FreezeV2-Re] STOP: FREEZEV2_CUDA_HOME has no executable bin/nvcc: $CUDA_HOME" >&2
            return 3
        fi
        current="$(_nvcc_minor_at "$nvcc_path")"
        if [[ "$current" != "$CUDA_VERSION" ]]; then
            echo "[FreezeV2-Re] STOP: FREEZEV2_CUDA_HOME provides nvcc ${current:-unknown}, expected $CUDA_VERSION." >&2
            return 3
        fi
        if ! _cuda_header_exists "$CUDA_HOME" "cusparse.h"; then
            echo "[FreezeV2-Re] STOP: FREEZEV2_CUDA_HOME is missing cusparse.h; a full CUDA development toolkit is required." >&2
            return 3
        fi
    else
        if ! _isolated_toolchain_ready; then
            echo "[FreezeV2-Re] preparing complete isolated CUDA $CUDA_VERSION build toolchain at $CUDA_TOOLKIT_PREFIX."
            mkdir -p "$(dirname "$CUDA_TOOLKIT_PREFIX")"

            if [[ -d "$CUDA_TOOLKIT_PREFIX/conda-meta" ]]; then
                conda install -y -p "$CUDA_TOOLKIT_PREFIX" \
                    --override-channels \
                    -c "nvidia/label/$CUDA_CONDA_LABEL" \
                    -c conda-forge \
                    "cuda-nvcc=${CUDA_VERSION}.*" \
                    "cuda-cudart-dev=${CUDA_VERSION}.*" \
                    "cuda-libraries-dev=${CUDA_VERSION}.*" \
                    "cuda-cccl=${CUDA_VERSION}.*" \
                    "gcc_linux-64=12.3.0" \
                    "gxx_linux-64=12.3.0"
            else
                rm -rf "$CUDA_TOOLKIT_PREFIX"
                conda create -y -p "$CUDA_TOOLKIT_PREFIX" \
                    --override-channels \
                    -c "nvidia/label/$CUDA_CONDA_LABEL" \
                    -c conda-forge \
                    "cuda-nvcc=${CUDA_VERSION}.*" \
                    "cuda-cudart-dev=${CUDA_VERSION}.*" \
                    "cuda-libraries-dev=${CUDA_VERSION}.*" \
                    "cuda-cccl=${CUDA_VERSION}.*" \
                    "gcc_linux-64=12.3.0" \
                    "gxx_linux-64=12.3.0"
            fi
        fi

        if ! _isolated_toolchain_ready; then
            echo "[FreezeV2-Re] STOP: isolated CUDA/GCC toolchain is incomplete at $CUDA_TOOLKIT_PREFIX." >&2
            return 3
        fi

        export CUDA_HOME="$CUDA_TOOLKIT_PREFIX"
        export CC="$CUDA_TOOLKIT_PREFIX/bin/x86_64-conda-linux-gnu-cc"
        export CXX="$CUDA_TOOLKIT_PREFIX/bin/x86_64-conda-linux-gnu-c++"
        export CUDAHOSTCXX="$CXX"
        export NVCC_CCBIN="$CC"
    fi

    export PATH="$CUDA_HOME/bin:$PATH"
    if [[ -d "$CUDA_HOME/targets/x86_64-linux/include" ]]; then
        export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
    else
        export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
    fi
    if [[ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]]; then
        export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    current="$(_nvcc_minor_at "$CUDA_HOME/bin/nvcc")"
    if [[ "$current" != "$CUDA_VERSION" ]]; then
        echo "[FreezeV2-Re] STOP: nvcc $CUDA_VERSION is required to build PointNet2 for $PROFILE." >&2
        echo "[FreezeV2-Re] current nvcc: ${current:-not found}" >&2
        echo "[FreezeV2-Re] set FREEZEV2_CUDA_HOME to a matching full CUDA toolkit if needed." >&2
        return 3
    fi

    echo "[FreezeV2-Re] CUDA_HOME: $CUDA_HOME"
    echo "[FreezeV2-Re] nvcc: $CUDA_HOME/bin/nvcc"
    echo "[FreezeV2-Re] nvcc CUDA: $current"
    if [[ -n "${CC:-}" ]]; then
        echo "[FreezeV2-Re] host C compiler: $CC"
    fi
    if [[ -n "${CXX:-}" ]]; then
        echo "[FreezeV2-Re] host C++ compiler: $CXX"
    fi
}

_pointnet_ready() {
    python - "$POINTNET_ARCH" <<'PY'
import sys
try:
    import torch
    import pointnet2_ops._ext
    from pointnet2_ops._version import __version__
    from pointnet2_ops.pointnet2_utils import furthest_point_sample

    expected = tuple(map(int, sys.argv[1].split(".")))
    if __version__ != "3.0.0":
        raise RuntimeError(__version__)
    if torch.cuda.get_device_capability(0) != expected:
        raise RuntimeError(torch.cuda.get_device_capability(0))
    xyz = torch.rand(1, 32, 3, device="cuda", dtype=torch.float32)
    idx = furthest_point_sample(xyz, 8)
    torch.cuda.synchronize()
    if tuple(idx.shape) != (1, 8) or not idx.is_cuda:
        raise RuntimeError("bad PointNet2 CUDA output")
except Exception as exc:
    print(f"[FreezeV2-Re] PointNet2 preflight failed: {exc}")
    raise SystemExit(1)
PY
}

_build_pointnet_from_gedi() {
    _prepare_cuda_toolkit

    local source_dir="$GEDI_ROOT/backbones/pointnet2_ops_lib"
    local build_dir
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/freezev2-pointnet2.XXXXXX")"
    cp -a "$source_dir/." "$build_dir/"

    python - "$build_dir/setup.py" "$POINTNET_ARCH" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
arch = sys.argv[2]
text = path.read_text()
pattern = r'os\.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=\s*[^\n]+'
replacement = f'os.environ["TORCH_CUDA_ARCH_LIST"] = "{arch}"'
new, count = re.subn(pattern, replacement, text, count=1)
if count != 1:
    raise SystemExit("could not patch GeDi PointNet2 TORCH_CUDA_ARCH_LIST")
path.write_text(new)
PY

    echo "[FreezeV2-Re] building PointNet2 $POINTNET2_VERSION from pinned GeDi source for sm_${POINTNET_ARCH/./}."
    _pip_clean install \
        --force-reinstall \
        --no-build-isolation \
        --no-deps \
        "$build_dir"

    rm -rf "$build_dir"
}

if ! _pointnet_ready; then
    if [[ -n "${FREEZEV2_POINTNET2_WHEEL:-}" ]]; then
        echo "[FreezeV2-Re] installing explicit PointNet2 wheel override: $FREEZEV2_POINTNET2_WHEEL"
        _pip_clean install --force-reinstall --no-deps "$FREEZEV2_POINTNET2_WHEEL"
    else
        _build_pointnet_from_gedi
    fi
fi

if ! _pointnet_ready; then
    echo "[FreezeV2-Re] STOP: PointNet2 CUDA smoke still fails for sm_${POINTNET_ARCH/./}." >&2
    exit 6
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
PY

# End-to-end smoke through the project adapter. This uses the official pinned
# GeDi network and checkpoint, the freshly verified PointNet2 CUDA extension,
# and freezev2's ABI-free CPU radius search.
python - "$GEDI_ROOT" "$GEDI_CHECKPOINT_CANONICAL" <<'PY'
from pathlib import Path
import sys

import numpy as np
import torch

from freezev2.gedi_bridge import GediExtractor

root = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
np.random.seed(0)
torch.manual_seed(0)

pcd = torch.rand(600, 3, dtype=torch.float32).numpy() * 0.1
pts = pcd[:2].copy()
extractor = GediExtractor(
    checkpoint=checkpoint,
    gedi_root=root,
    seed=0,
)
desc = extractor.encode(pts, pcd, object_diameter=1.0)

print("[FreezeV2-Re] GeDi descriptor smoke:", desc.shape, desc.dtype)
print("[FreezeV2-Re] GeDi descriptor finite:", bool(np.isfinite(desc).all()))
if desc.shape != (2, 64) or not np.isfinite(desc).all():
    raise SystemExit("GeDi end-to-end descriptor smoke failed")
PY

echo "[FreezeV2-Re] GeDi dependencies are ready for $PROFILE / sm_${POINTNET_ARCH/./}."
