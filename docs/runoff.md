# Integer-event payment runoff

`rustmc.DirichletMultinomialRunoff` fits incremental payment-event counts by origin
and development lag. It pools lag probabilities across cohorts, infers unknown
ultimate counts, and returns complete posterior payment paths. Currency amounts,
including amounts expressed in cents, are not independent events and must not be
used as count observations.

```python
import numpy as np
import rustmc

# Regular lags 0, 1, 2, followed by an unscheduled tail.
counts = np.array([
    [18., 9., 4., 1.],       # closed cohort
    [16., 8., 3., np.nan],   # all regular lags elapsed; tail remains open
    [12., 0., np.nan, np.nan],
    [7., np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan],  # a supplied future cohort
])
model = rustmc.DirichletMultinomialRunoff(
    alpha=[5., 3., 1.5, .5], total_shape=4., total_rate=.125,
)
fit = model.fit(
    counts, origins=[0, 1, 2, 3, 4], valuation=3,
    totals=[32, None, None, None, None],
    draws=1500, warmup=750, chains=4, seed=7,
)
calendar = fit.calendar_samples(3)  # [chain, draw, valuation+1 ... valuation+3]
tail = fit.tail_samples            # [chain, draw, cohort], no assigned date
print(np.quantile(calendar.sum(axis=-1), [.025, .5, .975]))
print(fit.summary())
```

## Data and time contract

The input matrix is a two-dimensional float64 NumPy array. Finite cells must be
nonnegative integers no larger than `2**53 - 1`; this bound also applies to each
cohort total. A literal zero is an observed zero. `NaN` means unobserved.
Input is incremental, not cumulative; difference cumulative histories before use.
Retained fit and calendar allocations are bounded at 25 million scalar values;
oversized requests raise errors. This is an allocation guard, not a byte-level
peak-memory guarantee, because nested arrays also carry metadata.

`origins` and `valuation` are integer periods on a common calendar. For example,
encode months as `12 * year + month - 1`. Lag zero falls in the origin period.
Every regular cell whose `origin + lag <= valuation` must be observed, including
zeros. Future regular cells must be `NaN`. Missing historical regular cells are
not supported; do not backfill them with zeros. Future cohorts may be supplied
with fully unobserved rows; the model does not create additional cohorts itself.

The last column is a **tail bucket for every lag at or beyond its index**.
It normally remains `NaN`. A numeric tail explicitly declares that the tail has
closed and its entire count is observed. It may close only once its first possible
period has arrived. A closed row with a known total must sum to that total.
An open tail does not become an observed zero merely because all regular lags
have elapsed. The model has no distribution over dates within the tail.

`totals` is a list with one integer or `None` per cohort. An integer conditions on
an externally known ultimate count and must cover all observed events. `None`
infers its ultimate. Omitting `totals` infers all ultimates. Known totals must be
actual conditioning information, not point estimates inserted in place of
uncertain accrual forecasts.

## Statistical model and inference

There is one shared stationary lag-probability vector:

```text
p ~ Dirichlet(alpha)
```

For a known ultimate `N_i`, a cohort's complete count vector has distribution
`Multinomial(N_i, p)`. Integrating out `p` produces Dirichlet-multinomial
allocations and dependence across cohorts. There is no separately estimated
cohort-specific probability vector or concentration. `alpha` is a fixed, proper
prior: its normalized values give prior lag means and its sum controls strength.

If all ultimates are known, prefix censoring permits exact conjugate inference.
With `q_l = p_l / sum(p[l:])`, prior hazards are independent
`Beta(alpha_l, sum(alpha[l+1:]))`. At each observed regular lag, a cohort contributes
its incremental count to the first shape and its known total minus cumulative
observed counts through that lag to the second shape. Future lag hazards receive
no exposure update. Draw these hazards independently, construct `p`, and allocate
each remaining count sequentially with binomial draws. With complete cohorts this
reduces to the usual `Dirichlet(alpha + column_sums)` posterior. `warmup` is ignored
for this exact path; `sampler` reports `independent_conjugate`.

Unknown ultimates instead have independent intensity priors:

```text
lambda_i ~ Gamma(total_shape, rate=total_rate)
x_i,l | lambda_i, p ~ Poisson(lambda_i * p_l), independently across lags
N_i = sum_l x_i,l
```

The prior ultimate mean is `total_shape / total_rate` and variance is
`total_shape / total_rate + total_shape / total_rate**2`. Constructor defaults
are shape 2 and rate 0.1, giving mean 20 and variance 220 events. Set these to a
credible cohort-scale prior; the defaults are not calibrated to a business domain.
All unknown cohorts use these same fixed hyperparameters, with independent
intensities conditional on them. No hierarchy over intensities is inferred.

Mixed known/unknown triangles use blocked latent-count Gibbs updates:

1. Given `p`, draw each unknown intensity from
   `Gamma(shape + observed_count, rate + observed_probability_mass)` and regenerate
   its missing independent Poisson counts. For known totals, regenerate missing
   counts by a multinomial allocation of the known remainder.
2. Draw `p ~ Dirichlet(alpha + all_completed_column_sums)`.

This targets the posterior of observed cells, integrating unknown totals rather
than normalizing incomplete rows to one. An observed zero contributes exposure;
a future cell does not. Fully unobserved cohorts retain their ultimate-count
prior marginal. All-zero cohorts remain usable under proper priors. Intensities,
completed allocations, and lag probabilities in each retained draw belong to the
same joint posterior. `sampler` reports `latent_count_gibbs`; choose adequate warmup
and retained draws and inspect diagnostics, especially with weak development data.

## Returned paths and aggregation

| Property or method | Shape and meaning |
| --- | --- |
| `allocation_samples` | `(chain, draw, cohort, lag)` complete integer counts, including observed cells and tail |
| `ultimate_samples` | `(chain, draw, cohort)` sums of complete allocations |
| `intensity_samples` | `(chain, draw, cohort)` Poisson intensity, NaN for known-total cohorts |
| `lag_probability_samples` | `(chain, draw, lag)` shared probabilities, including tail |
| `tail_samples` | `(chain, draw, cohort)` unobserved tail counts; closed observed tails return zero |
| `calendar_samples(steps)` | `(chain, draw, steps)` aggregate regular-lag events at valuation+1 through valuation+steps |
| `observed_mask` | `(cohort, lag)` true for observed cells |

Each complete row sums to its ultimate in every draw. Observed cells remain fixed.
Calendar aggregation uses the same draw across cohorts, retaining dependence
induced by the shared lag probabilities. The calendar excludes the tail and any
regular cells beyond the requested horizon. Sum **paths first**, then take
quantiles for totals or cumulative intervals; summing marginal interval endpoints
does not give an interval for the total. NumPy quantiles give equal-tailed
posterior-predictive intervals, not HDIs or confidence intervals.

`diagnostics()`, `summary()`, and `sampler_stats` reuse the common parameter
diagnostics. Coverage includes lag probabilities, unknown intensities, and unknown
ultimate counts. Acceptance rates and Hamiltonian divergences are unavailable for
both inference paths. Very short or constant traces have unavailable convergence
metrics. Diagnostics do not validate the model's payment assumptions.

## Limitations and evidence

Lag probabilities are stationary and shared across cohorts. There are no calendar
effects, cohort regressors, time-varying lags, refunds, negative counts, variable
cohort-specific total priors, or endogenous new-cohort forecasts. A tail timing
model and continuous monetary allocation model are separate extensions.

Event counts can be combined with an explicitly specified severity model to form
amounts. This API does not fit severity or infer dependence with a separate accrual
forecast. Do not multiply independently fitted posterior arrays and describe the
result as a joint model unless the independence and conditioning assumptions justify
that construction. In particular, simply aligning draw indices does not establish
posterior dependence between separate models.

Tests check complete-triangle Dirichlet moments, censored Beta-binomial predictions,
independent quadrature for an unknown-total posterior, Gamma-Poisson prior moments,
zero-versus-future masks, total conservation, seeded reproducibility, shared-lag
recovery, and a small rolling-valuation holdout against uniform remaining-lag
allocation. The holdout is a deterministic simulated regression test, not a general
calibration or performance claim. See `examples/payment_triangle_runoff.py` for an
end-to-end example with a conservation check.
