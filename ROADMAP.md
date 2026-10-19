# Direction

rustmc focuses on repeated Bayesian inference and structured partial pooling.
We want a small library whose supported models are easy to fit, inspect, and deploy.

The library has two inference paths, not one. Regression, calibration, and group
comparisons written with `ModelBuilder` run on the graph-based NUTS/HMC sampler. The
forecasting models — structural, seasonal, AR, dynamic GLM, hurdle, runoff, and the
Gaussian hierarchy — are hand-written samplers that do not use that graph, its
autodiff, or its samplers at all. Nor are they one kernel: Gibbs with FFBS for the
Gaussian state-space models (the local linear trend without `exog` integrates its
states out instead), exact conjugate draws for AR, block elliptical slice sampling
for dynamic GLMs, and for runoff either exact conjugate draws or latent-count Gibbs,
depending on whether every ultimate total is known.

What they share is worth stating precisely, because it decides which work pays off
twice. `diagnostics` is common to every model. The batch executor is shared inside
Rust. The Python result and prediction surface is common. `state_space` and
`forecast_diagnostics` are shared among the forecasting models only; the graph sampler
does not use them. So work on diagnostics, batching or the result surface reaches
every model, and work on a kernel reaches one.

These are the next five features, in dependency order. Each needs tests, documentation,
and a working example before it is complete.

| Feature | Work | Done when |
|---|---|---|
| Statistical release gates | Add fixed-data posterior references, repeated-simulation checks, and stricter positive recovery cases. Keep difficult negative controls separate. | Release checks fail on poor convergence, inaccurate posteriors, or silent diagnostic failures. Partly met in 0.14.0: the exactness gate runs NUTS at the shipped defaults, CI checks that it fails on two injected sampler faults, and the recovery suite runs at `target_accept=0.8`. |
| Repeated-inference benchmarks | Measure single and ragged repeated regressions, including setup, fitting, prediction, and memory. Reuse compiled models in competing engines. | Raw runs retain every failure and report useful throughput only when statistical quality passes. Met in 0.15.0 for Gaussian linear regression: `benchmarks/repeated.py` fits fixed-size and ragged datasets through one compiled model in rustmc (one fit per dataset, and one `sample_batch`), nutpie, NumPyro and PyMC, times setup, each fit and each prediction, records memory, and keeps failing and erroring fits, whose time counts and whose ESS does not. Other model families are not covered. |
| Native model artifacts | Move the current model definition, validation, and compilation into the Rust core. Keep Python as an adapter. | The same artifact can be loaded, bound, fitted, and evaluated from Python and Rust. Met for graph models and graph fits as of the third review; the forecasting models' artifacts are still defined per model. |
| Bounded batches | Stream inputs and results, choose retained outputs, and rerun jobs by stable ID. | Memory stays bounded as job count grows, and one failed job can be collected without losing completed work. Partly met in 0.14.0: `CompiledModel.sample_iter` streams `(id, data)` pairs a chunk at a time, keeps fits or summaries only, collects failures, and skips completed IDs on a rerun. Native memory over a long stream has not been measured. |
| Results and diagnostics | Share a result protocol with named dimensions and joint draw identity. Record energy, BFMI, termination reasons, and the actual algorithm. | Users can inspect generic and specialized fits consistently; inapplicable diagnostics remain unavailable. As of 0.14.0 HMC and NUTS record energy, E-BFMI and tree-depth saturation, and `warnings()` and `flagged()` apply one rule to every fit. As of 0.15.0 NUTS records why each tree stopped, every fit class with `get_samples()` or `get_samples_2d()` has both, the first with the chain and draw axes merged (`RunoffFit` has neither and keeps its own arrays), and every Bayesian forecast names its expected-observation draws `mean_samples` (the structural forecast's `*_paths` names are deprecated). Still missing: one result protocol with named dimensions and joint draw identity. |

## Next mathematical work

Expose the conjugate regression machinery already used by Gaussian AR models as an
ordinary regression API. Then add exact sufficient-statistic updates for that model.
Preserve its variance-scaled coefficient prior; other priors require other algorithms.

Evaluate one collapsed Gaussian hierarchy next: integrate out Gaussian effects, sample
hyperparameters, and reconstruct joint effects for prediction. Compare it with existing
Gibbs and noncentered sampling before extending the approach.

Learned scales for dynamic count and hurdle models can follow once the Gaussian path
is dependable. Their current scales and pooling strengths are fixed inputs.

`BayesianLocalLinearTrend` without `exog` no longer mixes slowly on its variances.
It now integrates the states out and slice-samples each variance given the other
two. On simulated series with 4 chains, 300 warmup and 500 draws, the level
variance's worst R-hat per setting went from 1.06–2.06 to at most 1.008. An
interweaving (ASIS) update of the old sampler was tried first; on the twelve series
of the new mixing test it raised the level variance's bulk ESS only to 13–181, and it
was not kept.

The trend with `exog`, the `StructuralModel` trend, the seasonal model and the local
level still draw their variances given sampled states. On 200-point series,
`benchmarks/llt_variance_mixing.py --other-models` shows the level variance of both
trends at a worst R-hat of 1.43–1.78 and a bulk ESS of 6–8. The seasonal variance
reaches a bulk ESS of only 10–35, and the local level's process variance 143–189. The
same approach should carry over to them, using the general filter's likelihood, and
so should the new test. The trend itself is slowest in the balanced regime at 1000
points and with a near-zero observation variance, where the level variance's
smallest bulk ESS is 388 to 479 out of 2000 draws. A joint update of the three
variances may help there.

## Scope

Keep the existing forecasting models and improve them through the shared foundations.
Add distributions or algorithms when a concrete supported workload needs them.
A universal inference planner, a tensor language, GPUs, distributed sampling, and
new volatility or regime-switching families are outside the current plan.

A release number follows evidence and compatibility, not feature count. The Rust API
and artifact compatibility policy will stabilize in stages.
