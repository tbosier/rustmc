# rustmc_core

`rustmc_core` is the Rust engine behind the [`rustmc`](https://pypi.org/project/rustmc/)
Python package. It provides Bayesian sampling, linear Gaussian state-space filtering and
smoothing, and fitted local-level, local-linear-trend, and autoregressive forecasting
models.

The Rust API is alpha and currently favors explicit model configuration over a broad
probabilistic-programming language. It is useful when inference or forecasting must run
inside a Rust process. It is not presented as a general replacement for Stan or PyMC.

```toml
[dependencies]
rustmc_core = "0.9"
```

```rust
use rustmc_core::state_space::LinearGaussianStateSpace;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LinearGaussianStateSpace::local_level(
        0.25, // process variance
        1.0,  // observation variance
        0.0,  // initial level mean
        10.0, // initial level variance
    )?;

    let filtered = model.filter(&[10.0, 10.5, 11.0, 10.8])?;
    println!("log likelihood: {}", filtered.log_likelihood);
    Ok(())
}
```

Forecast draws retain their joint path structure, allowing downstream code to calculate
distributions of cumulative values correctly. Missing observations are represented by
`NaN` in the state-space and local-level APIs.

Current limitations include no fitted seasonal or covariate state-space model, no stable
serialized deployment format, and no stability guarantee for the alpha Rust API. See the
[repository](https://github.com/tbosier/rustmc) for Python documentation, examples, and the
ordered roadmap.

The crates.io package named `rustmc` is unrelated. Use `rustmc_core` from Rust and
`rustmc` from PyPI/Python.
