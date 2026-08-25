# Environment switching

The reproduction intentionally uses two environments:

- `bop-eval`: BOP Toolkit, VisPy and user-space Mesa for headless CAD rendering/evaluation.
- `freeze`: CUDA/PyTorch, DINOv2 and later GeDi/pose inference.

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
