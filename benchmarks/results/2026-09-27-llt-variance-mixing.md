# Local-linear-trend variance mixing

One run of `benchmarks/llt_variance_mixing.py --other-models`, from the checkout at
revision `e9039d8c2d78da93bdd739ba3990e06bdfbc0958`, against two builds:

- **before**: the extension built from `d561103d9d70b1259bfcded6480c639da0d9650e`
  (0.13.0 plus release fixes), which draws the states by FFBS and then each variance
  given the states. Raw output: [before](2026-09-27-llt-variance-mixing-before.txt).
- **after**: the extension built from `50b8155` and `e9039d8` (identical Rust code),
  which integrates the states out and slice-samples each variance. Raw output:
  [after](2026-09-27-llt-variance-mixing-after.txt).

Both builds were made with `scripts/dev_pytest.sh` (`maturin build --release`) and
run as `PYTHONPATH=<build> python -m benchmarks.llt_variance_mixing --other-models`
on an AMD Ryzen 9 5900X (24 logical CPUs), Linux, Python 3.12.9, NumPy 2.4.2, with
`RAYON_NUM_THREADS` unset, so the four chains of each fit ran in parallel. Each raw
file begins with this environment as the script printed it.

Every fit uses 4 chains, 300 warmup and 500 retained draws, on simulated series of
200 and 1000 points, three data seeds each, in five regimes. The priors, starting
state and seeds are printed in the raw output. R-hat, ESS and posterior summaries are
deterministic for a build; seconds are from one run on one machine.

## Mixing

For each regime and length the table shows the worst split R-hat and the smallest
bulk ESS over the three seeds, out of 2000 draws, for the level variance and for the
worst of the three variances.

| Regime | n | Level R-hat before | after | Level bulk ESS before | after | Worst R-hat after, any variance | Smallest bulk ESS after, any variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 200 | 1.516 | 1.001 | 8 | 889 | 1.007 | 889 |
| balanced | 1000 | 2.056 | 1.008 | 6 | 388 | 1.008 | 388 |
| near-zero slope variance | 200 | 1.344 | 1.004 | 11 | 1351 | 1.005 | 1351 |
| near-zero slope variance | 1000 | 1.119 | 1.002 | 28 | 1164 | 1.006 | 1164 |
| near-zero level variance | 200 | 1.481 | 1.005 | 8 | 918 | 1.005 | 918 |
| near-zero level variance | 1000 | 1.774 | 1.002 | 6 | 1286 | 1.002 | 1286 |
| low signal-to-noise | 200 | 1.470 | 1.007 | 8 | 1592 | 1.007 | 1538 |
| low signal-to-noise | 1000 | 1.727 | 1.003 | 6 | 1275 | 1.003 | 1275 |
| near-zero observation variance | 200 | 1.321 | 1.007 | 10 | 460 | 1.007 | 460 |
| near-zero observation variance | 1000 | 1.062 | 1.006 | 58 | 479 | 1.006 | 479 |

Before, the slope variance's worst R-hat per setting was 1.051 to 1.421 and the
observation variance's 1.005 to 1.299. After, no variance's R-hat exceeded 1.008 in
any setting.

## Cost

A fit took 2.0 to 2.3 times as long (median over seeds, per setting): each sweep
runs the Kalman filter once per slice-sampling evaluation instead of FFBS once. The
level variance's median bulk ESS per second rose by a factor of 3.7 to 96 across
the ten settings. In the low signal-to-noise regime, where the old sampler already
mixed the observation variance (worst R-hat 1.005 and 1.028), that variance's ESS per
second fell to 0.68 (n = 200) and 0.57 (n = 1000) of its old value.

## Same posterior

On a 40-point series with one missing value, both builds were run for 4 × 25,000
draws with two seeds, beside posterior means from quadrature on a 40-point-per-axis
grid in log variance, using the general `LinearGaussianStateSpace` filter. Every
sampled mean of the three variances and the terminal level and slope is within 1.8
Monte Carlo standard errors of the quadrature mean for the new build, and within 1.3
for the old build. The new build's bulk ESS there is 32,541 to 93,709 against 2,104 to
40,585 for the old.

## Recovery

The `cover` column counts the seeds whose central 90% interval contains the true
variance. Coverage is incomplete in both builds, and similar: of the 90 (setting,
variance, seed) intervals, 52 covered the truth before and 53 after. Of the 37 misses
after, 30 have one of two causes, neither of them the sampler:

- True variances of 1e-6 and 1e-4 sit far below where the inverse-gamma priors
  (shape 2, scales 0.01, 0.001 and 0.1) put mass, so those posteriors are pulled up
  to the prior's scale: for example a median of 7.5e-3 for a true level variance of
  1e-6 at n = 200. None of those 18 intervals covered the truth in either build.
- A true level variance of 0.05 is underestimated in the near-zero slope and
  near-zero observation settings, at both lengths: none of those 12 intervals covered
  it, and even at n = 1000 the medians are 0.034 and 0.035. To check
  that this is the posterior, the first seed of each of those two settings was
  refitted with 2000 draws per chain and compared with quadrature on a 36-point-per-
  axis grid in log variance, six posterior standard deviations either side of the
  sampled mean, using the general filter.
  Sampled and quadrature means agreed: 0.03456 (MCSE 0.00012) against 0.03461, and
  0.03464 (MCSE 0.00007) against 0.03464. That check is not in the script.

The other seven misses are one or two out of three seeds in a setting, which 90%
intervals can produce by chance.

At n = 200 the old sampler's medians often differ from the new ones (0.0055 against
0.0073 for the balanced level variance), because the old chains had not converged.

## Samplers not changed

The same series (n = 200, balanced and near-zero level variance) fitted with the
samplers that still draw variances given sampled states give identical numbers under
both builds. The trend with `exog` and the `StructuralModel` trend reach a level
variance R-hat of 1.43 to 1.78 and a bulk ESS of 6 to 8; the seasonal model's seasonal
variance a bulk ESS of 10 to 35; the local level's process variance 143 to 189.

## Not measured here

An interweaving (ASIS) update added to the old sampler was tried first, with a
prototype that was not kept. On the twelve 200-point series of the Rust test
`state_variances_mix_in_every_signal_to_noise_regime` (not the series above), it
raised the level variance's bulk ESS from 6–15 to 13–181; R-hat stayed as high as
1.25. There is no retained output for it.
