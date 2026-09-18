#!/usr/bin/env bash
# Build this worktree's extension into an isolated directory and run pytest
# against it, without disturbing the shared .venv used by other worktrees.
#
#   ./scripts/dev_pytest.sh [pytest args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/home/taylo/projects/rustmc/.venv
SCRATCH="${TMPDIR:-/tmp}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/.target}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uvcache}"
mkdir -p "$UV_CACHE_DIR"
# rustup finds its toolchain through HOME, and this script redirects HOME
# below so the test run cannot write to the real cache. Pin rustup/cargo to
# the real home first, so an inherited HOME override does not break the build
# with "rustup could not choose a version of cargo to run".
export RUSTUP_HOME="${RUSTUP_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)/.cargo}"
rm -rf "$ROOT/.pybuild" "$ROOT/.wheel"
"$VENV/bin/maturin" build --release --manifest-path "$ROOT/python_bindings/Cargo.toml" --out "$ROOT/.wheel" >/dev/null
"$VENV/bin/python" -m zipfile -e "$(ls "$ROOT"/.wheel/*.whl | head -1)" "$ROOT/.pybuild"
export PYTHONPATH="$ROOT/.pybuild"
export XDG_CACHE_HOME="$SCRATCH/c" MPLCONFIGDIR="$SCRATCH/mpl" HOME="$SCRATCH/h"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HOME"
actual="$("$VENV/bin/python" -c 'import rustmc; print(rustmc.__file__)')"
case "$actual" in
  "$ROOT"/.pybuild/*) ;;
  *) echo "FATAL: imported $actual, not this worktree's build" >&2; exit 1 ;;
esac
cd "$ROOT"
exec "$VENV/bin/python" -m pytest "$@"
