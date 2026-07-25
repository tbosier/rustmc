"""
rustmc vs PyMC — Heavy Multivariate Regression Benchmark
=========================================================

10-parameter multiple regression with 100K observations and 8 chains.

This workload stresses the inner sampling loop: each gradient evaluation
requires summing over 100K data points for each of 10 parameters, and the
8 chains highlight thread-based (rustmc) vs process-based (PyMC)
parallelism.

Fairness notes (previously violated by this script — see
benchmarks/RESULTS_TEMPLATE.md and the PR description for the full audit):
  - PyMC previously ran with its (slow) default NUTS sampler and no
    explicit tune count or seed. It now runs with an explicit seed and
    tune=1000 (matching rustmc's warmup=1000) using BOTH PyMC's default
    NUTS and the nutpie backend, so the README doesn't quietly compare
    rustmc against the weakest available PyMC configuration.
  - sigma=0.5 is the true simulated noise, fixed (not estimated) on every
    engine — same parameter count (10 betas) everywhere.
  - Only wall time and point estimates were reported before; R-hat,
    divergences, and ESS/s are now reported for every engine, and only
    possible because both sides use >=2 chains.
"""

import time

import numpy as np

from bench_common import PhaseTimer, print_environment, peak_rss_mb

# ---------------------------------------------------------------------------
# Synthetic data: y = X @ beta + noise
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
N = 100_000
P = 10
TRUE_SIGMA = 0.5

X = rng.standard_normal((N, P))
beta_true = np.linspace(0.5, 3.0, P)
y = X @ beta_true + rng.standard_normal(N) * TRUE_SIGMA

NUM_CHAINS = 8
NUM_DRAWS = 2000
WARMUP = 1000

print(f"Model: {P} parameters, {N:,} observations, {NUM_CHAINS} chains, "
      f"{WARMUP} warmup + {NUM_DRAWS} draws, seed={SEED}")
print(f"True betas: {beta_true.tolist()}")
print(f"True sigma: {TRUE_SIGMA} (fixed, not estimated, on every engine)")
print()
print_environment()
print()

results = []

# ---------------------------------------------------------------------------
# Fit with rustmc
# ---------------------------------------------------------------------------
import rustmc as rmc

pt = PhaseTimer()
with pt.phase("build"):
    data = {f"x_{i}": np.ascontiguousarray(X[:, i]) for i in range(P)}
    data["y"] = y
    builder = rmc.ModelBuilder(data=data)
    params = [builder.normal_prior(f"beta_{i}", mu=0.0, sigma=10.0) for i in range(P)]
    mu_expr = params[0] * "x_0"
    for i in range(1, P):
        mu_expr = mu_expr + params[i] * f"x_{i}"
    builder.normal_likelihood("obs", mu_expr=mu_expr, sigma=TRUE_SIGMA, observed_key="y")
    model = builder.build()

with pt.phase("compile+sample"):
    fit = rmc.sample(
        model_spec=model,
        chains=NUM_CHAINS,
        draws=NUM_DRAWS,
        warmup=WARMUP,
        seed=SEED,
        show_progress=False,
    )

with pt.phase("postprocess"):
    diag = fit.diagnostics()
    beta_ess = np.mean([d["ess_bulk"] for d in diag if d["name"].startswith("beta_")])
    max_rhat = max(d["r_hat"] for d in diag if d["name"].startswith("beta_"))
    rustmc_means = fit.mean()
    beta_est = np.array([rustmc_means[f"beta_{i}"] for i in range(P)])
    rmse = float(np.sqrt(np.mean((beta_est - beta_true) ** 2)))

divs = sum(fit.divergences())
print("rustmc")
pt.report()
print(f"  Divergences : {divs}")
print(f"  Beta ESS    : {beta_ess:.0f}  (mean across {P} params)")
print(f"  Max R-hat   : {max_rhat:.4f}")
print(f"  ESS/s       : {beta_ess / pt.total:.1f}")
print(f"  Beta RMSE   : {rmse:.4f}")
print(f"  Peak RSS    : {peak_rss_mb():.0f} MB")
for i in [0, 1, P - 1]:
    print(f"  beta_{i}: true={beta_true[i]:.3f}, est={beta_est[i]:.3f}")
print()
results.append(("rustmc", pt.total, beta_ess, max_rhat, divs, rmse))

# ---------------------------------------------------------------------------
# Fit with PyMC — both default NUTS and nutpie, explicit seed both times
# ---------------------------------------------------------------------------
def run_pymc(nuts_sampler):
    import pymc as pm
    import arviz as az

    pt = PhaseTimer()
    with pt.phase("build"):
        with pm.Model() as pymc_model:
            betas = [pm.Normal(f"beta_{i}", 0, 10) for i in range(P)]
            mu = sum(betas[i] * X[:, i] for i in range(P))
            pm.Normal("obs", mu=mu, sigma=TRUE_SIGMA, observed=y)

    with pt.phase("compile+sample"):
        with pymc_model:
            trace = pm.sample(
                NUM_DRAWS,
                tune=WARMUP,
                chains=NUM_CHAINS,
                cores=NUM_CHAINS,
                nuts_sampler=nuts_sampler,
                random_seed=SEED,
                progressbar=False,
            )

    with pt.phase("postprocess"):
        beta_names = [f"beta_{i}" for i in range(P)]
        ess_vals = np.array([float(az.ess(trace)[n].values) for n in beta_names])
        rhat_vals = np.array([float(az.rhat(trace)[n].values) for n in beta_names])
        beta_est = np.array([float(trace.posterior[n].mean()) for n in beta_names])
        rmse = float(np.sqrt(np.mean((beta_est - beta_true) ** 2)))

    divs = int(trace.sample_stats["diverging"].values.sum())
    print(f"PyMC ({nuts_sampler})")
    pt.report()
    print(f"  Divergences : {divs}")
    print(f"  Beta ESS    : {ess_vals.mean():.0f}  (mean across {P} params)")
    print(f"  Max R-hat   : {rhat_vals.max():.4f}")
    print(f"  ESS/s       : {ess_vals.mean() / pt.total:.1f}")
    print(f"  Beta RMSE   : {rmse:.4f}")
    print(f"  Peak RSS    : {peak_rss_mb():.0f} MB (parent process; PyMC forks {NUM_CHAINS} workers)")
    for i in [0, 1, P - 1]:
        print(f"  beta_{i}: true={beta_true[i]:.3f}, est={beta_est[i]:.3f}")
    print()
    return (f"PyMC ({nuts_sampler})", pt.total, ess_vals.mean(), rhat_vals.max(), divs, rmse)


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
print(f"{'Engine':<16}{'Time(s)':>10}{'BetaESS':>10}{'ESS/s':>10}{'MaxR-hat':>10}{'Divs':>7}{'BetaRMSE':>10}")
print("-" * 100)
for name, t, ess, rhat, divs, rmse in results:
    print(f"{name:<16}{t:>10.2f}{ess:>10.0f}{ess/t:>10.1f}{rhat:>10.4f}{divs:>7}{rmse:>10.4f}")

if len(results) > 1:
    base = results[0]
    for other in results[1:]:
        speedup = (base[2] / base[1]) / (other[2] / other[1])
        direction = "faster" if speedup >= 1 else "slower"
        factor = speedup if speedup >= 1 else 1 / speedup
        print(f"{base[0]} is {factor:.2f}x {direction} than {other[0]} by ESS/s")
