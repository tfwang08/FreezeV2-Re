#!/usr/bin/env bash

# Source this file to enter the CPU BOP/VisPy environment with user-space Mesa.
# Usage:
#   source scripts/use_bop_eval.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source scripts/use_bop_eval.sh" >&2
    exit 1
fi

_freezev2_load_conda() {
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

_freezev2_strip_path_entry() {
    local value="$1"
    local remove="$2"
    local out=""
    local entry
    local old_ifs="$IFS"
    IFS=':'
    for entry in $value; do
        if [[ -n "$entry" && "$entry" != "$remove" ]]; then
            out="${out:+$out:}$entry"
        fi
    done
    IFS="$old_ifs"
    printf '%s' "$out"
}

_freezev2_load_conda || return 1
conda activate bop-eval || return 1

export MESA_ROOT="${FREEZEV2_MESA_ROOT:-$HOME/.local/mesa-runtime/usr}"
export MESA_LIB="$MESA_ROOT/lib/x86_64-linux-gnu"

if [[ ! -f "$MESA_LIB/libEGL_mesa.so.0" ]]; then
    echo "Missing Mesa EGL runtime: $MESA_LIB/libEGL_mesa.so.0" >&2
    echo "Set FREEZEV2_MESA_ROOT if your user-space Mesa is elsewhere." >&2
    return 1
fi
if [[ ! -f "$MESA_LIB/dri/swrast_dri.so" ]]; then
    echo "Missing Mesa software renderer: $MESA_LIB/dri/swrast_dri.so" >&2
    return 1
fi
if [[ ! -f "$MESA_ROOT/share/glvnd/egl_vendor.d/50_mesa.json" ]]; then
    echo "Missing Mesa EGL vendor JSON under $MESA_ROOT/share/glvnd/egl_vendor.d" >&2
    return 1
fi

_clean_ld="$(_freezev2_strip_path_entry "${LD_LIBRARY_PATH:-}" "$MESA_LIB")"
export LD_LIBRARY_PATH="$MESA_LIB${_clean_ld:+:$_clean_ld}"
unset _clean_ld

export LIBGL_DRIVERS_PATH="$MESA_LIB/dri"
export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_ROOT/share/glvnd/egl_vendor.d/50_mesa.json"
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export PYOPENGL_PLATFORM=egl

unset -f _freezev2_load_conda _freezev2_strip_path_entry

echo "[FreezeV2-Re] environment: bop-eval"
echo "[FreezeV2-Re] Mesa: $MESA_ROOT"
echo "[FreezeV2-Re] EGL: surfaceless software rendering"
