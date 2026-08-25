#!/usr/bin/env bash

# Source this file to enter the isolated GeDi CUDA environment.
# Usage:
#   source scripts/use_gedi.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced: source scripts/use_gedi.sh" >&2
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
conda activate gedi || return 1

_mesa_root="${MESA_ROOT:-${FREEZEV2_MESA_ROOT:-$HOME/.local/mesa-runtime/usr}}"
_mesa_lib="${MESA_LIB:-$_mesa_root/lib/x86_64-linux-gnu}"
_clean_ld="$(_freezev2_strip_path_entry "${LD_LIBRARY_PATH:-}" "$_mesa_lib")"
if [[ -n "$_clean_ld" ]]; then
    export LD_LIBRARY_PATH="$_clean_ld"
else
    unset LD_LIBRARY_PATH
fi

unset MESA_ROOT MESA_LIB
unset LIBGL_DRIVERS_PATH
unset __EGL_VENDOR_LIBRARY_FILENAMES
unset EGL_PLATFORM
unset LIBGL_ALWAYS_SOFTWARE
unset PYOPENGL_PLATFORM
unset GALLIUM_DRIVER
unset EGL_LOG_LEVEL
unset LIBGL_DEBUG

unset _mesa_root _mesa_lib _clean_ld
unset -f _freezev2_load_conda _freezev2_strip_path_entry

echo "[FreezeV2-Re] environment: gedi"
echo "[FreezeV2-Re] removed user-space Mesa/EGL overrides"
