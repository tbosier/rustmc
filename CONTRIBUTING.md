# Contributing to rustmc

rustmc welcomes focused contributions that improve correctness, repeated inference,
partial pooling, and deployment.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin numpy pytest
maturin develop --manifest-path python_bindings/Cargo.toml --release
```

`rustmc_core` supports Rust 1.87 and newer (`rust-version` in `rust_core/Cargo.toml`);
CI builds it with exactly that toolchain, so raise the field in the same pull request
as code that needs a newer compiler.

To test a linked git worktree without installing into the shared virtualenv,
`./scripts/dev_pytest.sh -q` builds that worktree's extension into `.pybuild/` and runs
pytest against it. It uses `$RUSTMC_VENV` if set, otherwise the main checkout's
`.venv`.

Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --release
python -m pytest -q

# CI runs these too, and they fail for reasons the tests above will not catch.
python3 scripts/verify_version.py             # manifests, binding dep, Cargo.lock agree
python scripts/run_examples.py --timeout 180  # every example in examples/README.md runs
python scripts/build_example_docs.py --check  # docs pages still match the code
```

The last one covers three things: `docs/examples/*.md` is generated from the example
of the same name and must not have drifted, the code blocks on each guide page must
still run when read in order, and the output shown on the landing page must be what
the code above it prints. If it reports a generated page as stale, run
`python scripts/build_example_docs.py` and commit the result. The other two it
reports are for you to fix by hand.

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

## Documentation

Lead with what the reader can do. Use short paragraphs, familiar words, and runnable
examples. Cut repeated claims and promotional language. Keep limitations next to
the behavior they qualify. See [Google’s technical writing guide](https://developers.google.com/style/tone)
and [GOV.UK’s plain-language guidance](https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/).
