# Posterior checks, 11 September 2026

All 76 attempted fits passed. Four analytic reference cases ran
under three seeds, followed by 64 independently simulated Gaussian regressions.

| Metric | Observed |
|---|---:|
| Maximum R-hat | 1.00306 |
| Minimum bulk ESS | 2295.9 |
| Minimum tail ESS | 1709.1 |
| Divergences | 0 |
| Maximum mean error / reference SD | 0.0400 |
| Maximum standardized covariance error | 0.0558 |

Central 90% parameter coverage was [0.953125, 0.90625]. With 64 independent
replicates, this is a limited-power check. It covers these models under correct
specification, not every supported family or misspecified data.

Source revision: `554f972e91c3cc9a041e5a7a6aec84d835001022`. The native extension hash, controls, every attempt,
and reference moments are in [the raw report](2026-09-11-posteriors.json).
This is a statistical check, not a performance comparison.

```bash
python -m benchmarks.validate_posteriors --replicates 64 --output /tmp/posteriors.json
```
