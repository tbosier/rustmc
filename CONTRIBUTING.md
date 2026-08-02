# Contributing to rustmc

rustmc welcomes focused contributions that improve correctness, forecasting quality,
reproducibility, or deployment ergonomics.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin numpy pytest
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
python -m pytest -q
```

## Evidence expectations

- A log-density or gradient change needs a finite-difference, analytic, or independent
  reference test.
- A sampler change needs a seeded recovery test and relevant diagnostic assertions.
- A forecasting change needs a known-data-generating-process test and, when applicable,
  rolling-origin evaluation against a simple baseline.
- A performance claim needs retained raw output, revision, environment, exact command,
  matched work, and statistical-quality metrics. Use
  [`benchmarks/RESULTS_TEMPLATE.md`](benchmarks/RESULTS_TEMPLATE.md).
- A public API change needs Python tests and documentation in the same pull request.

Do not describe equal-tailed intervals as HDIs, posterior-predictive intervals as
confidence intervals, or a successful convergence diagnostic as proof that a model is
correct. State limitations and negative benchmark results plainly.

## Pull requests

Keep each pull request narrow enough to review. Include:

- the problem and intended behavior;
- tests or independent evidence;
- commands actually run;
- compatibility or migration effects; and
- remaining limitations.

Generated files, local worktrees, editor logs, and internal review notes do not belong in
the repository.
