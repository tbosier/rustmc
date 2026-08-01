# Validation results

Snapshot from 2026-07-31. This records commands actually run; it is not a claim that
the entire roadmap or every statistical family has been certified.

## Focused recovery correction

`ridge_regression_recovers_high_dimensional_coefficients` previously flattened a
column-oriented fixture and passed it to `Graph::store_matrix`, whose contract is
row-major. The test now stores `X[row, col]` in row-major order, uses a six-parameter
seeded problem, and checks coefficient RMSE at 0.20 instead of the old 0.95 tolerance.

```text
cargo test -p rustmc_core --test recovery_suite \
  ridge_regression_recovers_high_dimensional_coefficients --release -- --nocapture
result: 1 passed; sampler test time 0.10s
```

The prior malformed version was reported to take roughly 135 seconds and did not test
the matrix it claimed to test. The corrected focused case is both materially stricter
and suitable for routine CI.

## Python source and wheel checks

After a release editable build, the default Python suite reported:

```text
python -m pytest tests -q
61 passed, 4 skipped, 1 deselected
```

The skipped packaging tests require a current-version wheel in `dist/` or
`target/wheels/`; stale ignored wheels are deliberately not selected. CI separately
builds a wheel, installs it in a clean environment outside the checkout, and runs the
same import/API/NumPy/end-to-end sampling verification for the supported Python 3.9–3.13
matrix. No claim is made here about an unsupported interpreter or an artifact whose
build provenance was not retained.

## Static checks

`cargo fmt --all -- --check` passes and is enforced in CI. Strict
`cargo clippy --workspace --all-targets -- -D warnings` still fails on existing style
lint debt (principally `needless_range_loop` and large internal function signatures),
so a knowingly red strict-clippy release gate was not added.

## Benchmark interpretation

The benchmark harness can record environment, phase timing, R-hat, ESS, divergences,
posterior error, and memory where available. No numeric benchmark results are checked in
because the raw command output needed to audit earlier figures was not retained. The
suite must be rerun with raw logs and exact revision provenance before making throughput,
compile-once reuse, scaling, or cross-engine speed claims.
