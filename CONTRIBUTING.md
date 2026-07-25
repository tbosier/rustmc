# Contributing to rustmc

## Prerequisites

Building rustmc from source requires:

- **Rust** 1.75+ (workspace uses the 2021 edition; developed against 1.89).
  Install via [rustup](https://rustup.rs/).
- **Python** 3.9–3.13 (CPython; the classifiers in `pyproject.toml` list
  the versions we build and test wheels for). Any of these works to build
  and develop against.
- **maturin** `>=1.0,<2.0` — builds the PyO3 extension module and wheels.
  Install into your virtualenv with `pip install maturin` or `uv pip
  install maturin`.
- **numpy** — runtime dependency, also needed to run the examples/tests.
- A C linker (`cc`/`clang`/`gcc`) available on `PATH`, as with any Rust
  build. No other system libraries are required — `faer` (linear algebra)
  is pure Rust with no BLAS/LAPACK dependency.

No macOS- or Windows-specific system packages are known to be required,
but only Linux builds have been locally verified as part of packaging
validation (see `.github/workflows/ci.yml` and the notes below). Treat
macOS/Windows as unverified until someone builds and tests there.

## Building from source

```bash
git clone https://github.com/tbosier/rustmc.git
cd rustmc

# Rust core (no Python involved)
cargo test --workspace

# Python extension, editable install into a venv
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release --manifest-path python_bindings/Cargo.toml
```

## Building a wheel

```bash
pip install maturin
maturin build --release --manifest-path python_bindings/Cargo.toml --out dist
pip install dist/rustmc-*.whl
```

## Running the test suite

Rust:

```bash
cargo test --all
```

Python (pytest suite under `tests/`):

```bash
pip install -e ".[test]"     # or: pip install pytest
.venv/bin/python -m pytest tests/ -q
```

By default the Python suite skips tests that need a pre-built wheel
(`tests/test_packaging.py`'s file-content/metadata checks — build a wheel
first with `maturin build` if you want those to run instead of skip) and
deselects the `network` marker, which builds a wheel from scratch and
installs it into a fresh virtualenv end-to-end
(`pytest -m network tests/test_packaging.py` to run it explicitly).

## Verifying a wheel install manually

`scripts/verify_wheel_install.py` is a small, dependency-free script that
imports rustmc, checks the public API surface, and runs a tiny end-to-end
sampling run. Run it with the interpreter of a clean environment (not this
repo's dev `.venv`) after installing a built wheel there, from a working
directory outside the repo checkout, to prove you're exercising the
installed wheel rather than any local source:

```bash
python -m venv /tmp/rustmc-clean && /tmp/rustmc-clean/bin/pip install dist/rustmc-*.whl numpy
cp scripts/verify_wheel_install.py /tmp/rustmc-clean/
(cd /tmp && /tmp/rustmc-clean/bin/python /tmp/rustmc-clean/verify_wheel_install.py)
```

It exits non-zero and prints a `FAIL:` line on any problem, including if
`rustmc.__file__` doesn't resolve to a `site-packages` directory.

## Known gaps / follow-ups

- No `py.typed` marker or `.pyi` type stubs are currently shipped, so type
  checkers (mypy/pyright) treat the package as untyped
  (`tests/test_packaging.py::test_wheel_ships_type_information` documents
  this as an `xfail`).
- CI (`.github/workflows/ci.yml`) currently only builds/tests on Linux
  (`x86_64-unknown-linux-gnu`). macOS and Windows wheel-build jobs already
  exist for tagged releases but have no install/import/sampling
  verification job — add matrix entries there once someone can verify the
  steps on those platforms (this repo's dev sandbox is Linux-only).
- `cargo fmt --check` and `cargo clippy -- -D warnings` are not enforced
  in CI yet; the current tree has pre-existing formatting/clippy issues
  being addressed in a separate pass. Do not add a blocking gate for
  these until that cleanup lands.
