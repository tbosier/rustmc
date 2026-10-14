# Statistical release checks

This page describes the gate that has to pass before a release: a fixed set of models
whose exact posteriors are known analytically, refitted and compared against them. It
is for contributors and for anyone judging how much the samplers are checked. You do
not need it to use the library.

Run `python -m benchmarks.validate_posteriors --output /tmp/posteriors.json` after
building rustmc. The command returns a nonzero status if a fit or gate fails.

Four fixed datasets have analytic posterior moments: Normal location, correlated
Gaussian regression, Beta–Bernoulli, and Gamma–Poisson. Each runs under three seeds.
The run fails if any of those cases does not produce its three records, or if a case
outside that list is attempted; the report's `case_coverage` block records what ran
against what was required.

Checks require R-hat <= 1.01, bulk and tail ESS >= 400, no divergences, and bounded
errors in posterior means and covariance. Every parameter is checked separately,
against the full range each diagnostic can take rather than only against its threshold.
A missing, non-finite, non-numeric, or out-of-range R-hat or ESS for any single
parameter fails the run on its own and names that parameter in the report; the bounds
are derived from the estimators themselves, so a value outside them did not come from a
fit. Divergence telemetry must be one whole non-negative count per chain, since a sum of
counts can cancel and cannot show a missing chain. Failed values are kept in the report,
written as JSON null where they are not finite. These checks compare with the posterior
for the observed dataset, rather than only with the parameter that generated the data.

A separate experiment draws regression coefficients from the stated prior, simulates
data with NumPy, and refits each dataset. It retains parameter coverage and rank
histograms, every attempted fit, and numerical or convergence failures. Coverage uses
independent replicate counts and a stated conservative finite-sample bound. Rank
histograms are exploratory because the retained MCMC draws are autocorrelated.
A small experiment has limited power; passing is not proof of calibration for every model.

The JSON records source revision, native module hash, versions, command, seeds, and
reference values. CI runs 32 replicates and uploads the report even if a gate fails,
including failures while building a case, failures while simulating a replicate, and
diagnostics that cannot be serialized. No report is written if importing `rustmc`
fails, or if the trailing `git rev-parse` and native-hash calls raise.
The default local run uses 64. Increase the replicate count for a more precise study.

The coverage check passes when each coefficient's 90% interval coverage lies within
`sqrt(ln(4000) / (2n))` of the nominal 0.90, where `n` is the replicate count. At the
32 replicates CI runs, that accepts any coverage from 0.54 to 1.00, so the per-PR run
is a smoke check that catches only gross miscalibration. At the local default of 64 the accepted band is about 0.65 to 1.00, and
256 replicates narrow it to about 0.77 to 1.00. The band reaches 1.00 at every one of
these counts, so over-wide intervals are not caught at all. The report records the
tolerance it used as `coverage_tolerance`.

Neither gate samples at the shipped default `target_accept=0.8`. The fixed
reference cases above run at `target_accept=0.95`, and the Rust recovery tests in
`rust_core/tests/recovery_suite.rs` run every case at 0.999.

`rust_core/tests/analytic_posteriors.rs` does run at the defaults. It samples a
100-dimensional independent Gaussian and a 50-dimensional correlated one with NUTS,
and requires every coordinate's bulk ESS to reach 400 and its mean and variance to lie
within four Monte Carlo standard errors of the exact values. The same checks on static
HMC are kept as ignored tests, because at the default 15 leapfrog steps HMC fails
them: its integration time lands near a full period of these targets, and from 4000
draws one coordinate's bulk ESS is 7.

Rust recovery tests enforce the same convergence thresholds for positive cases.
The Gaussian partial-pooling case uses noncentered effects. Centered funnel and
centered eight-schools cases remain negative controls: poor recovery must produce a
diagnostic signal. None of these checks establishes calibration under misspecification.
