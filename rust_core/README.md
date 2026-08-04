# rustmc_core

`rustmc_core` is the Rust engine behind the [`rustmc`](https://pypi.org/project/rustmc/)
Python package. It combines graph-based Bayesian sampling with specialized algorithms
for model structures that admit more direct inference, including conjugate and linear
Gaussian state-space methods.

The Rust API is alpha and currently favors explicit model configuration over a broad
probabilistic-programming language. It is useful when inference must run inside a Rust
process. It is not presented as a general replacement for Stan or PyMC.

```toml
[dependencies]
rustmc_core = "0.10"
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

Fixed-system forecasts expose the joint future-observation covariance and cumulative
Gaussian moments. Posterior-predictive draws from fitted models retain their joint path
structure, allowing downstream code to calculate distributions of cumulative values
correctly. Missing observations are represented by `NaN` in state-space APIs.

For related ragged series, `hierarchical::fit_hierarchical_mean` fits one joint
population → group → program Gaussian posterior with a specialized conjugate Gibbs
kernel. Its forecast paths are indexed `[chain][draw][program][step]`, so downstream
Rust code can aggregate aligned draws without discarding cross-program dependence.

Current limitations include no fitted covariate or multiple-seasonality state-space
model, no stable serialized deployment format, and no stability guarantee for the alpha Rust API. See the
[repository](https://github.com/tbosier/rustmc) for Python documentation, examples, and the
ordered roadmap.

The crates.io package named `rustmc` is unrelated. Use `rustmc_core` from Rust and
`rustmc` from PyPI/Python.
