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

## GeDi binary stack

GeDi setup is **prebuilt-only**. `scripts/setup_gedi.sh` does not compile Open3D or PointNet2 and has no source/JIT fallback.

The `freeze` environment is intentionally pinned to:

```text
Python       3.11
PyTorch      2.2.2+cu121
torchvision  0.17.2+cu121
torchaudio   2.2.2+cu121
NumPy        1.26.4
Open3D       0.19.0
PointNet2    3.0.0
```

Python 3.11 is required by the matching prebuilt PointNet2 wheel. The selected wheel was built with PyTorch 2.2.2 / CUDA 12.1 and includes `sm_80`; its PointNet2 CUDA/C++ sources match the ops vendored by the pinned GeDi repository.

Run:

```bash
source scripts/use_freeze.sh
bash scripts/setup_gedi.sh
```

On the first run the setup script may change the existing `freeze` environment from Python 3.12 to Python 3.11, then reinstall the released binary stack. Later runs skip that work when the exact versions already pass their smoke tests.

The default PointNet2 wheel is:

```text
https://github.com/YanWenKun/ComfyUI-3D-Pack-LinuxWheels/releases/download/v2/pointnet2_ops-3.0.0-cp311-cp311-linux_x86_64.whl
```

Override it with a local copy when needed:

```bash
export FREEZEV2_POINTNET2_WHEEL=/path/to/pointnet2_ops-3.0.0-cp311-cp311-linux_x86_64.whl
bash scripts/setup_gedi.sh
```

The official GeDi checkpoint is a local input because cluster nodes may not reach Google Drive. By default it must be at:

```text
external/gedi/data/chkpts/3dmatch/chkpt.tar
```

or can be supplied with:

```bash
export FREEZEV2_GEDI_CHECKPOINT=/path/to/chkpt.tar
```

Setup verifies the checkpoint key, Open3D CUDA/`radius_search`, a PointNet2 CUDA kernel, and finally computes a small 32-D GeDi descriptor batch end to end.
