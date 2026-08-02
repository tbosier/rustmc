"""
rustmc vs PyMC — 500-parameter linear regression, matched protocol
===================================================================
Same model, same data, same draws/warmup/chains/seed on every engine.
Reports phase-separated timing (build / sample+compile / postprocess),
ESS/s, max R-hat, divergences, and posterior RMSE against the known
simulated beta — not wall time alone.

Fairness notes:
  - sigma=1.0 is the true simulated noise, fixed (not estimated) on every
    engine — same parameter count (intercept + 500 betas) everywhere.
  - PyMC is run with both the explicit ``pymc`` and ``nutpie`` backends,
    so the backend choice is visible rather than described as a default.
  - chains=2 and seed=42 on every engine.
"""
import time
import warnings
import logging
import numpy as np
import arviz as az

from bench_common import PhaseTimer, print_environment, peak_rss_mb

# ── Config ────────────────────────────────────────────────────────────────
N_OBS    = 2_000
N_PARAMS = 500
DRAWS    = 500
WARMUP   = 500
CHAINS   = 2  # >=2 chains so split R-hat is defined on both sides
SEED     = 42
TRUE_SIGMA = 1.0

# ── Data ──────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
true_beta  = rng.standard_normal(N_PARAMS) * 0.1
X          = rng.standard_normal((N_OBS, N_PARAMS))
y          = X @ true_beta + rng.standard_normal(N_OBS) * TRUE_SIGMA

print(f"Model : {N_OBS:,} obs, {N_PARAMS:,} params, true sigma={TRUE_SIGMA} (fixed on both sides)")
print(f"Run   : {WARMUP} warmup + {DRAWS} draws, {CHAINS} chain(s), seed={SEED}")
print()
print_environment()
print()

results = []

# ── rustmc ────────────────────────────────────────────────────────────────
print("=" * 50)
print("rustmc (faer MatVecMul + Rayon)")
print("=" * 50)

import rustmc as rmc

pt = PhaseTimer()
with pt.phase("build"):
    builder   = rmc.ModelBuilder(data={"X": X, "y": y})
    intercept = builder.normal_prior("intercept", mu=0.0, sigma=10.0)
    beta      = builder.vector_normal_prior("beta", n=N_PARAMS, mu=0.0, sigma=1.0)
    builder.normal_likelihood("obs",
        mu_expr=intercept + beta @ "X",
        sigma=TRUE_SIGMA,
        observed_key="y")
    model = builder.build()

with pt.phase("compile+sample"):
    rmc_result = rmc.sample(model,
                            draws=DRAWS, warmup=WARMUP,
                            chains=CHAINS, seed=SEED,
                            show_progress=False)

with pt.phase("postprocess"):
    rmc_samples = rmc_result.get_samples_2d()
    beta_names = sorted(
        (name for name in rmc_samples if name.startswith("beta[")),
        key=lambda name: int(name.removeprefix("beta[").removesuffix("]")),
    )
    rmc_beta_ess_values = np.asarray(
        [float(az.ess(rmc_samples[name])) for name in beta_names]
    )
    rmc_rhat_values = np.asarray(
        [float(az.rhat(rmc_samples[name])) for name in beta_names]
    )
    rmc_beta_ess = float(rmc_beta_ess_values.mean())
    rmc_max_rhat = float(rmc_rhat_values.max())
    beta_means = np.asarray([rmc_samples[name].mean() for name in beta_names])
    rmc_rmse = float(np.sqrt(np.mean((beta_means - true_beta) ** 2)))

rmc_divs = sum(rmc_result.divergences())
pt.report()
print(f"  Accept rate : {rmc_result.accept_rates()[0]:.3f}")
print(f"  Divergences : {rmc_divs}")
print(f"  Beta ESS    : {rmc_beta_ess:.0f}  (mean across {N_PARAMS} params)")
print(f"  Max R-hat   : {rmc_max_rhat:.4f}")
print(f"  ESS/s       : {rmc_beta_ess / pt.total:.1f}")
print(f"  Beta RMSE   : {rmc_rmse:.4f}  (vs. known true beta)")
print(f"  Peak RSS    : {peak_rss_mb():.0f} MB")
results.append(("rustmc", pt.total, rmc_beta_ess, rmc_max_rhat, rmc_divs, rmc_rmse))

# ── PyMC ──────────────────────────────────────────────────────────────────
logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pymc as pm


def run_pymc(nuts_sampler: str):
    pt = PhaseTimer()
    with pt.phase("build"):
        with pm.Model() as pymc_model:
            pm_intercept = pm.Normal("intercept", mu=0.0, sigma=10.0)
            pm_beta      = pm.Normal("beta", mu=0.0, sigma=1.0, shape=N_PARAMS)
            mu           = pm_intercept + pm.math.dot(X, pm_beta)
            pm.Normal("obs", mu=mu, sigma=TRUE_SIGMA, observed=y)

    with pt.phase("compile+sample"):
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
        beta_ess = az.ess(trace)["beta"].values
        beta_rhat = az.rhat(trace)["beta"].values
        pymc_beta_ess = float(beta_ess.mean())
        pymc_max_rhat = float(beta_rhat.max())
        beta_means = trace.posterior["beta"].values.reshape(-1, N_PARAMS).mean(axis=0)
        pymc_rmse = float(np.sqrt(np.mean((beta_means - true_beta) ** 2)))

    divs = int(trace.sample_stats["diverging"].values.sum())

    print()
    print("=" * 50)
    print(f"PyMC ({nuts_sampler})")
    print("=" * 50)
    pt.report()
    print(f"  Divergences : {divs}")
    print(f"  Beta ESS    : {pymc_beta_ess:.0f}  (mean across {N_PARAMS} params)")
    print(f"  Max R-hat   : {pymc_max_rhat:.4f}")
    print(f"  ESS/s       : {pymc_beta_ess / pt.total:.1f}")
    print(f"  Beta RMSE   : {pymc_rmse:.4f}  (vs. known true beta)")
    print(f"  Peak RSS    : {peak_rss_mb():.0f} MB (parent process; PyMC may fork workers)")
    return (f"PyMC ({nuts_sampler})", pt.total, pymc_beta_ess, pymc_max_rhat, divs, pymc_rmse)


for backend in ("pymc", "nutpie"):
    try:
        results.append(run_pymc(backend))
    except ImportError as e:
        print(f"\nPyMC backend '{backend}' unavailable — skipping: {e}")
    except Exception as e:
        print(f"\nPyMC backend '{backend}' failed — skipping: {e}")

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("Summary")
print("=" * 50)
print(f"{'Engine':<16}{'Time(s)':>10}{'BetaESS':>10}{'ESS/s':>10}{'MaxR-hat':>10}{'Divs':>7}{'BetaRMSE':>10}")
for name, t, ess, rhat, divs, rmse in results:
    print(f"{name:<16}{t:>10.2f}{ess:>10.0f}{ess/t:>10.1f}{rhat:>10.4f}{divs:>7}{rmse:>10.4f}")

if len(results) > 1:
    base = results[0]
    for other in results[1:]:
        speedup = (base[2] / base[1]) / (other[2] / other[1])
        direction = "faster" if speedup >= 1 else "slower"
        factor = speedup if speedup >= 1 else 1 / speedup
        print(f"\n{base[0]} is {factor:.2f}x {direction} than {other[0]} by ESS/s")
