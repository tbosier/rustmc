"""
rustmc vs PyMC — simple linear regression, matched protocol
============================================================

Same model, same data, same chains/warmup/draws, same seed on both sides.
Reports phase-separated timing (build / compile / sample) plus posterior
quality (R-hat, bulk ESS, ESS/s, divergences) and posterior error against
the known simulated beta — not wall time alone.

Fairness notes (see benchmarks/RESULTS_TEMPLATE.md for the full audit):
  - sigma=1.0 is the true simulated noise and is fixed (not estimated) on
    BOTH sides — same number of estimated parameters (1: beta).
  - chains, tune/warmup, draws, and seed are identical on both sides.
  - PyMC is run twice: once with its own default NUTS, once with the
    nutpie (Rust/JAX) backend, so neither a pessimistic nor an optimistic
    PyMC baseline is silently chosen.

Requirements:
    pip install pymc nutpie numpy arviz
"""
import time

import numpy as np

from bench_common import PhaseTimer, print_environment, peak_rss_mb

# ---------------------------------------------------------------------------
# Config — identical for every engine below
# ---------------------------------------------------------------------------
SEED = 42
N = 10_000
TRUE_BETA = 2.5
TRUE_SIGMA = 1.0  # known to both engines; neither estimates it
CHAINS = 4
DRAWS = 1000
WARMUP = 500

rng = np.random.default_rng(SEED)
x = rng.standard_normal(N)
y = TRUE_BETA * x + rng.standard_normal(N) * TRUE_SIGMA
data = {"x": x, "y": y}

print(f"Model : y = beta*x + N(0, {TRUE_SIGMA}), N={N:,} obs, true beta={TRUE_BETA}")
print(f"Run   : {CHAINS} chains, {WARMUP} warmup + {DRAWS} draws, seed={SEED}")
print()
print_environment()
print()

results = []

# ---------------------------------------------------------------------------
# rustmc
# ---------------------------------------------------------------------------
import rustmc as rmc

pt = PhaseTimer()
with pt.phase("build"):
    builder = rmc.ModelBuilder(data=data)
    beta = builder.normal_prior("beta", mu=0.0, sigma=1.0)
    mu_expr = beta * "x"
    builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=TRUE_SIGMA, observed_key="y")
    model = builder.build()

with pt.phase("sample"):  # rustmc compiles the Evaluator lazily inside sample()
    fit = rmc.sample(
        model_spec=model,
        chains=CHAINS,
        draws=DRAWS,
        warmup=WARMUP,
        seed=SEED,
        show_progress=False,
    )

with pt.phase("postprocess"):
    diag = fit.diagnostics()
    beta_diag = next(d for d in diag if d["name"] == "beta")
    rustmc_mean = fit.mean()["beta"]
    rustmc_std = fit.std()["beta"]
    rustmc_rmse = abs(rustmc_mean - TRUE_BETA)

rustmc_ess = beta_diag["ess_bulk"]
rustmc_rhat = beta_diag["r_hat"]
rustmc_divs = sum(fit.divergences())

print("rustmc")
pt.report()
print(f"  beta mean/std   : {rustmc_mean:.4f} +/- {rustmc_std:.4f}  (true {TRUE_BETA})")
print(f"  R-hat / ESS_bulk: {rustmc_rhat:.4f} / {rustmc_ess:.0f}")
print(f"  Divergences     : {rustmc_divs}")
print(f"  ESS/s           : {rustmc_ess / pt.total:.1f}")
print(f"  Peak RSS (MB)   : {peak_rss_mb():.0f}")
print()

results.append(("rustmc", pt.total, rustmc_ess, rustmc_rhat, rustmc_divs, rustmc_rmse))


# ---------------------------------------------------------------------------
# PyMC helper — runs once per backend so both a pessimistic (default) and
# an optimistic (nutpie) PyMC baseline are reported, not just whichever
# makes the comparison look best.
# ---------------------------------------------------------------------------
def run_pymc(nuts_sampler: str):
    import pymc as pm
    import arviz as az

    pt = PhaseTimer()
    with pt.phase("build"):
        with pm.Model() as pymc_model:
            beta_rv = pm.Normal("beta", 0, 1)
            mu = beta_rv * x
            pm.Normal("obs", mu=mu, sigma=TRUE_SIGMA, observed=y)

    with pt.phase("compile+sample"):
        # PyMC/nutpie compile the sampling function as part of pm.sample();
        # there is no public hook to separate compile from sample time here,
        # so both phases are reported together and labeled as such.
        with pymc_model:
            trace = pm.sample(
                draws=DRAWS,
                tune=WARMUP,
                chains=CHAINS,
                nuts_sampler=nuts_sampler,
                progressbar=False,
                random_seed=SEED,
            )

    with pt.phase("postprocess"):
        ess = float(az.ess(trace)["beta"].values)
        rhat = float(az.rhat(trace)["beta"].values)
        divs = int(trace.sample_stats["diverging"].values.sum())
        mean = float(trace.posterior["beta"].mean())
        rmse = abs(mean - TRUE_BETA)

    label = f"PyMC ({nuts_sampler})"
    print(label)
    pt.report()
    print(f"  beta mean       : {mean:.4f}  (true {TRUE_BETA})")
    print(f"  R-hat / ESS_bulk: {rhat:.4f} / {ess:.0f}")
    print(f"  Divergences     : {divs}")
    print(f"  ESS/s           : {ess / pt.total:.1f}")
    print(f"  Peak RSS (MB)   : {peak_rss_mb():.0f}  (parent process only; PyMC/nutpie may fork workers)")
    print()
    return ("PyMC (" + nuts_sampler + ")", pt.total, ess, rhat, divs, rmse)


for backend in ("pymc", "nutpie"):
    try:
        results.append(run_pymc(backend))
    except ImportError as e:
        print(f"PyMC backend '{backend}' unavailable — skipping: {e}\n")
    except Exception as e:
        print(f"PyMC backend '{backend}' failed — skipping: {e}\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 100)
print(f"{'Engine':<18}{'Time(s)':>10}{'ESS_bulk':>11}{'ESS/s':>11}{'R-hat':>9}{'Divs':>7}{'|beta_err|':>12}")
print("-" * 100)
for name, t, ess, rhat, divs, rmse in results:
    print(f"{name:<18}{t:>10.2f}{ess:>11.0f}{ess/t:>11.1f}{rhat:>9.4f}{divs:>7}{rmse:>12.5f}")
print()
print("Note: R-hat needs >=2 chains to be meaningful; both sides use "
      f"{CHAINS} chains here so it is comparable.")
