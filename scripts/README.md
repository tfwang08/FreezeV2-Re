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

GeDi setup is **prebuilt-only**. `scripts/setup_gedi.sh` does not compile Open3D or PointNet2.

The released Open3D 0.19.0 wheel requires PyTorch 2.2.*, so the `freeze` environment is intentionally pinned to:

```text
Python       3.12
PyTorch      2.2.2+cu121
torchvision  0.17.2+cu121
torchaudio   2.2.2+cu121
NumPy        1.26.4
Open3D       0.19.0
```

Run:

```bash
source scripts/use_freeze.sh
bash scripts/setup_gedi.sh
```

The script installs the released PyTorch/Open3D wheels and verifies that `open3d.ml.torch.ops.radius_search` loads with CUDA.

Official GeDi vendors a PointNet2 CUDA extension but does not ship a matching prebuilt wheel for the pinned Python/PyTorch/CUDA stack. The setup script therefore stops instead of compiling it. Once an exact compatible wheel is identified, pass it explicitly:

```bash
export FREEZEV2_POINTNET2_WHEEL=/path/to/pointnet2_ops.whl
bash scripts/setup_gedi.sh
```

The supplied PointNet2 wheel is accepted only after its CUDA extension imports and executes a CUDA smoke test.
