# rustmc_core

Version 0.13 closes a correctness review and carries breaking changes to the alpha Rust
API. `nuts::run_chain`, `nuts::run_chain_bound`, `hmc::run_chain` and
`hmc::run_chain_bound` now return `Result<ChainResult, String>` and reject discrete
latent parameters, which they previously accepted without a guard. A new
`graph::Op::BoundedSigmoid` variant breaks any exhaustive match on `Op`, and reverse
mode now calls `ElementwiseOp::adjoints` rather than `derivatives`. `GraphModel` gained
`sample_prior` and `prior_predictive`. See the
[changelog](https://github.com/tbosier/rustmc/blob/main/CHANGELOG.md).

Version 0.12 added `structural` for composable Gaussian/Student-t state-space models,
`dynamic_glm` for joint count/hurdle/pooled dynamic inference, and
`target::{LogDensity, sample_target}` for native custom unconstrained densities and
gradients. Generic batch options share the stable-ID forecast executor. Graph observation
simulation and expression dimensions are shared across fitting and prediction.
The Rust API remains pre-1.0; see the repository model guides for kernel assumptions.

`rustmc_core` is the Rust engine behind the [`rustmc`](https://pypi.org/project/rustmc/)
Python package. It combines graph-based Bayesian sampling with specialized algorithms
for model structures that admit more direct inference, including conjugate and linear
Gaussian state-space methods. The two are genuinely separate: the forecasting modules
(`structural`, `bayesian_*`, `dynamic_glm`, `hurdle`, `runoff`, `hierarchical`) do not
use `graph`, `autodiff`, `nuts` or `hmc` at all. `state_space` and
`forecast_diagnostics` are shared among those modules but are not used by the graph
sampler either. The two paths genuinely share `diagnostics`, and `forecast_batch`,
which `sampler` imports directly rather than through the binding layer.

The Rust API is alpha and currently favors explicit model configuration over a broad
probabilistic-programming language. It is useful when inference must run inside a Rust
process. It is not presented as a general replacement for Stan or PyMC.

```toml
[dependencies]
rustmc_core = "0.13"
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
kernel. `HierarchicalMeanForecast::observation_paths` is indexed
`[chain][draw][program * horizon + step]`: every program's whole path is contiguous
inside one per-draw vector, not a separate allocation per program. `state_means` is
indexed `[chain][draw][program]`, because the latent level is static over the horizon.
Both share a draw axis, so downstream Rust code can aggregate aligned draws without
discarding cross-program dependence.

`bayesian_regression` jointly samples Gaussian coefficient, structural-state, and
variance uncertainty with time-varying designs. `forecast_batch` provides independent
cell execution with stable ID seeds. `hurdle` fits sparse nonnegative amounts, and
`runoff` provides censored payment-count development with known or uncertain ultimates.

Current limitations include no automatic inference planner, no stability guarantee for the alpha Rust API. See the
[repository](https://github.com/tbosier/rustmc) for Python documentation, examples, and the
ordered roadmap.

The crates.io package named `rustmc` is unrelated. Use `rustmc_core` from Rust and
`rustmc` from PyPI/Python.

Load Python-authored graph artifacts through `model::GraphModel`; see `examples/load_model.rs` and the repository’s native-models guide.
