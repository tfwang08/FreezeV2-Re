# Environment switching

The reproduction intentionally uses two environments:

- `bop-eval`: BOP Toolkit, VisPy and user-space Mesa for headless CAD rendering/evaluation.
- `freeze`: CUDA/PyTorch, DINOv2 and GeDi/pose inference.

Always **source** the switch scripts so they can modify the current shell:

```bash
source scripts/use_bop_eval.sh
```

This activates `bop-eval` and configures the user-space Mesa runtime at:

```text
$HOME/.local/mesa-runtime/usr
```

Override that location when needed:

```bash
export FREEZEV2_MESA_ROOT=/path/to/mesa-runtime/usr
source scripts/use_bop_eval.sh
```

Switch back to the CUDA environment with:

```bash
source scripts/use_freeze.sh
```

`use_freeze.sh` removes the Mesa/EGL overrides, including the project Mesa entry from `LD_LIBRARY_PATH`, before CUDA/PyTorch work continues.

Do not execute these scripts as child processes (`bash scripts/use_*.sh`); environment changes from a child shell cannot affect the current terminal.

## GeDi GPU stack

`scripts/setup_gedi.sh` selects a reproducible stack from the active NVIDIA GPU compute capability and builds PointNet2 from the exact CUDA/C++ source vendored by the pinned GeDi commit. It does **not** trust a generic prebuilt PointNet2 wheel by default, because such a wheel may contain cubins for an unrelated GPU architecture.

Supported profiles are:

| GPU profile | Compute capability | Python | PyTorch | CUDA | PointNet2 target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ampere A800/A100 | 8.0 | 3.11 | 2.2.2 | 12.1 | `sm_80` |
| RTX 50-series Blackwell | 12.0 | 3.11 | 2.7.1 | 12.8 | `sm_120` |

NumPy remains pinned to 1.26.4. The Blackwell profile uses a CUDA-12.8 PyTorch build because older CUDA PyTorch binaries do not contain Blackwell kernels.

GeDi itself remains the pinned official source/checkpoint. Its only Open3D-ML use is fixed-radius neighbour lookup. `freezev2.gedi_bridge` supplies an equivalent CPU torch radius search at that API boundary, so GeDi is no longer coupled to Open3D 0.19's PyTorch-2.2 binary ABI. The PointNet2 network, checkpoint and descriptor computation remain the official GeDi implementation.

Run:

```bash
source scripts/use_freeze.sh
bash scripts/setup_gedi.sh
```

Setup does the following:

1. Detects a single active GPU compute capability (override with `FREEZEV2_GPU_CC=8.0` or `12.0`).
2. Installs the matching Python/PyTorch/CUDA runtime profile when needed.
3. Checks out the pinned GeDi revision and validates the official checkpoint.
4. Ensures a matching `nvcc` toolkit is available, installing the compiler/dev pieces into the `freeze` conda environment when needed.
5. Builds `pointnet2_ops==3.0.0` from GeDi's vendored source for the exact target architecture.
6. Executes a real CUDA farthest-point-sampling kernel.
7. Computes a finite two-scale 64-D GeDi descriptor batch end to end through `freezev2.gedi_bridge`.

The official GeDi checkpoint is a local input because cluster nodes may not reach Google Drive. By default it must be at:

```text
external/gedi/data/chkpts/3dmatch/chkpt.tar
```

or can be supplied with:

```bash
export FREEZEV2_GEDI_CHECKPOINT=/path/to/chkpt.tar
```

### Overrides

Use a local CUDA toolkit instead of the conda-installed compiler:

```bash
export FREEZEV2_CUDA_HOME=/path/to/cuda-12.1   # A800/A100
# or /path/to/cuda-12.8 for RTX 50-series
bash scripts/setup_gedi.sh
```

For mirrors or air-gapped nodes, override the PyTorch index:

```bash
export FREEZEV2_PYTORCH_INDEX_URL=https://your-mirror/pytorch-wheels/cu128
bash scripts/setup_gedi.sh
```

A known-good architecture-compatible PointNet2 wheel can still be supplied explicitly, but there is no generic default wheel:

```bash
export FREEZEV2_POINTNET2_WHEEL=/path/to/pointnet2_ops-3.0.0-....whl
bash scripts/setup_gedi.sh
```

The supplied wheel must pass the real CUDA kernel smoke on the active GPU or setup stops.
