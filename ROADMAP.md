# Direction

rustmc focuses on repeated Bayesian inference and structured partial pooling.
Regression, calibration, group comparisons, and forecasting share the same core.
We want a small library whose supported models are easy to fit, inspect, and deploy.

These are the next five features, in dependency order. Each needs tests, documentation,
and a working example before it is complete.

| Feature | Work | Done when |
|---|---|---|
| Statistical release gates | Add fixed-data posterior references, repeated-simulation checks, and stricter positive recovery cases. Keep difficult negative controls separate. | Release checks fail on poor convergence, inaccurate posteriors, or silent diagnostic failures. |
| Repeated-inference benchmarks | Measure single and ragged repeated regressions, including setup, fitting, prediction, and memory. Reuse compiled models in competing engines. | Raw runs retain every failure and report useful throughput only when statistical quality passes. |
| Native model artifacts | Move the current model definition, validation, and compilation into the Rust core. Keep Python as an adapter. | The same artifact can be loaded, bound, fitted, and evaluated from Python and Rust. |
| Bounded batches | Stream inputs and results, choose retained outputs, and rerun jobs by stable ID. | Memory stays bounded as job count grows, and one failed job can be collected without losing completed work. |
| Results and diagnostics | Share a result protocol with named dimensions and joint draw identity. Record energy, BFMI, termination reasons, and the actual algorithm. | Users can inspect generic and specialized fits consistently; inapplicable diagnostics remain unavailable. |

## Next mathematical work

Expose the conjugate regression machinery already used by Gaussian AR models as an
ordinary regression API. Then add exact sufficient-statistic updates for that model.
Preserve its variance-scaled coefficient prior; other priors require other algorithms.

Evaluate one collapsed Gaussian hierarchy next: integrate out Gaussian effects, sample
hyperparameters, and reconstruct joint effects for prediction. Compare it with existing
Gibbs and noncentered sampling before extending the approach.

Learned scales for dynamic count and hurdle models can follow once the Gaussian path
is dependable. Their current scales and pooling strengths are fixed inputs.

## Scope

Keep the existing forecasting models and improve them through the shared foundations.
Add distributions or algorithms when a concrete supported workload needs them.
A universal inference planner, a tensor language, GPUs, distributed sampling, and
new volatility or regime-switching families are outside the current plan.

A release number follows evidence and compatibility, not feature count. The Rust API
and artifact compatibility policy will stabilize in stages.
