# Independent forecasting batches and diagnostics

`BayesianLocalLevel`, `BayesianSeasonalLocalLevel`, `BayesianLocalLinearTrend`,
and `BayesianAR` expose native independent-cell fitting:

```python
import numpy as np
import rustmc as rmc

prior = rmc.InverseGammaPrior(3.0, 0.4)
model = rmc.BayesianLocalLevel(
    process_variance_prior=prior, observation_variance_prior=prior,
)
batch = model.fit_batch(
    [np.array([0.2, 0.4, np.nan, 0.7]), np.array([1.0, 0.9, 1.2])],
    ids=["north", "south"],
    chains=4, draws=500, warmup=250, seed=42,
    threads=4, chunk_size=32, errors="collect",
)
print(batch.ids, batch.errors)
print(batch["north"].summary())
forecast = batch.forecast(12, seed=43, threads=4, errors="collect")
paths = forecast["north"].observation_samples  # chain × draw × horizon
```

`ids` are required, unique, exact UTF-8 strings; returned order matches input order.
`models=[model_for_north, model_for_south]` supplies per-cell priors and structural
configuration, including different seasonal periods or AR orders. An entry of
`None` uses the calling model. Histories may be ragged. Missing observations retain
their time positions for the state-space models; AR continues to require finite
histories. Sampling controls are shared across the call. AR uses exact independent
conjugate draws and ignores batch `warmup`/`thin`.

`errors="collect"` returns successful fits alongside a mapping of cell ID to error
text. Both validation and numerical errors are isolated; failed entries in `.results`
are `None`, and indexing a failed ID raises its error. `errors="raise"` stops
scheduling work after an observed failure; already-running workers may finish.
Duplicate IDs, wrong batch lengths, or invalid execution options always raise.
`batch.diagnostics()` maps successful IDs to parameter reports and failed IDs to
`None`. Forecast errors follow the same contract, including previous fit failures.

One private Rayon pool serves the entire call, including nested chain work. `threads`
and `chunk_size` must be positive; the default worker limit is one. Native execution
releases the GIL. `chunk_size` bounds cells dispatched simultaneously; it **does not**
bound retained output memory. Fit objects retain all posterior parameter draws, and
forecast objects retain all joint paths. For bounded retention, submit slices of a
larger workload, persist their summaries or draws, then release each batch before
submitting the next slice. Accessing `.results` or an individual fit/forecast creates
an owned copy. Avoid keeping both copies for large workloads.

Seeds depend on `(seed, cell_id, domain)` followed by each sampler's existing
model/domain/chain derivation, never cell position or worker scheduling. The cell
encoding is version-one FNV-1a over ASCII `rustmc-cell-v1`, the 64-bit little-endian
seed, the 64-bit little-endian UTF-8 ID byte length, ID bytes, and domain bytes
(`fit` or `forecast`). `forecast_cell_seed(seed, cell_id, domain="fit")` exposes this
contract for single-cell reproduction. Keep seeds and IDs unchanged when reordering,
chunking, subsetting, or resuming. Different chains and cells use distinct streams;
64-bit hashes have the usual theoretical collision possibility. IDs are not Unicode
normalized. Exact numerical reproducibility is guaranteed for the same build and
platform; floating-point libraries can differ across platforms.

Batch forecasts preserve each model's paired posterior parameters and coherent future
path. Aggregation across cells represents independent series and introduces no shared
shocks or hierarchical dependence.

All four fitted models and hierarchical Gibbs fits expose `summary()`, `diagnostics()`,
and a `sampler_stats` dictionary. Parameter reports reuse rank-normalized folded split
R-hat, bulk/tail ESS, mean MCSE, and 94% empirical HDIs. These HDIs summarize parameter
draws; forecast intervals retain their existing equal-tailed semantics. Local and trend
reports cover learned variances and retained terminal states. Seasonal reports include
**every** terminal seasonal state coordinate; AR reports cover all coefficients and
innovation variance. Historical latent states are not retained by these models.

Gibbs/FFBS and exact conjugate sampling have no Hamiltonian divergences or Metropolis
acceptance rate: both statistics are `None`. `numerical_failures=0` describes a successful
fit; numerical failures terminate the affected cell and appear separately in batch
errors. Nonfinite, constant, or fewer-than-six-draw traces have unavailable convergence
metrics (`None` in Python). Infinite R-hat is retained when it detects separated constant
chains. A one-chain fit can have split diagnostics, but
`independent_chain_comparison=False` makes clear that it cannot compare independent
chains. Finite diagnostics do not prove convergence, identification, or model adequacy.
