# Statistical release checks

Run `python -m benchmarks.validate_posteriors --output /tmp/posteriors.json` after
building rustmc. The command returns a nonzero status if a fit or gate fails.

Four fixed datasets have analytic posterior moments: Normal location, correlated
Gaussian regression, Beta–Bernoulli, and Gamma–Poisson. Each runs under three seeds.
Checks require R-hat <= 1.01, bulk and tail ESS >= 400, no divergences, and bounded
errors in posterior means and covariance. These compare with the posterior for the
observed dataset, rather than only with the parameter that generated the data.

A separate experiment draws regression coefficients from the stated prior, simulates
data with NumPy, and refits each dataset. It retains parameter coverage and rank
histograms, every attempted fit, and numerical or convergence failures. Coverage uses
independent replicate counts and a stated conservative finite-sample bound. Rank
histograms are exploratory because the retained MCMC draws are autocorrelated.
A small experiment has limited power; passing is not proof of calibration for every model.

The JSON records source revision, native module hash, versions, command, seeds, and
reference values. CI runs 32 replicates and uploads the report even if a gate fails.
The default local run uses 64. Increase the replicate count for a more precise study.

Rust recovery tests enforce the same convergence thresholds for positive cases.
The Gaussian partial-pooling case uses noncentered effects. Centered funnel and
centered eight-schools cases remain negative controls: poor recovery must produce a
diagnostic signal. None of these checks establishes calibration under misspecification.
