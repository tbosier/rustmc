# rustmc benchmark result

Copy this file to `benchmarks/results/YYYY-MM-DD-<short-description>.md`,
fill it in from an actual run, and commit it alongside the raw stdout log
(or attach the log). Never fill in numbers you didn't actually run — if a
benchmark could not be executed in your environment, say so explicitly in
the relevant field instead of leaving it blank or guessing.

## Environment

| Field | Value |
|---|---|
| Date | |
| CPU model | |
| Logical CPUs | |
| `RAYON_NUM_THREADS` | |
| `OMP_NUM_THREADS` | |
| OS | |
| rustmc version | |
| PyMC version | |
| nutpie version | |
| NumPyro version | |
| JAX version / backend / devices | |
| arviz version | |
| numpy version | |
| Python version | |

## Model / workload

- Script: `benchmarks/run.py`
- Machine-readable config file:
- Data SHA-256 (must match across every engine subprocess):
- Description of the model (parameters, likelihood, observation count):
- True/simulated parameter values (if synthetic data):
- Which parameters are estimated vs. fixed as known constants, **on each
  engine being compared** (this is the #1 place unfair comparisons hide —
  e.g. handing the true noise sigma to one engine as a constant while the
  other estimates it):
- Chains / warmup (tune) / draws — must be identical across engines being
  compared, or the difference must be called out and justified:
- Seed(s) used, and whether every engine got an explicit, fixed seed:
- Thread/process count used by each engine (Rayon threads, PyMC
  `cores=`, BLAS threads, etc.):

## Timing (phase-separated — never report one conflated number)

| Phase | Engine A | Engine B | ... |
|---|---|---|---|
| Model construction | | | |
| Compilation (only where separately observable) | | | |
| Warmup / adaptation (only where separately observable) | | | |
| Retained sampling (only where separately observable) | | | |
| Engine-native combined phase(s), named exactly | | | |
| Post-processing (diagnostics, export) | | | |
| **Total wall time** | | | |

Note: some engines (e.g. PyMC with the nutpie or numpyro backend) do not
expose a public hook to separate compilation from sampling; when phases
are combined, label the combined phase explicitly rather than attributing
the whole time to "sampling".

## Statistical quality (never report speed without this)

| Metric | Engine A | Engine B | ... |
|---|---|---|---|
| Divergences | | | |
| Max rank-normalized folded split R-hat (needs >=2 chains) | | | |
| Mean / min bulk ESS | | | |
| Mean bulk ESS / fit-second (primary cross-engine throughput) | | | |
| Mean bulk ESS / retained-sampling-second (only where separately observable) | | | |
| Posterior error vs. known simulated truth (e.g. RMSE of posterior mean vs. true parameter) | | | |
| Peak RSS (MB) | | | |
| Machine-readable quality gate passed? (necessary, not sufficient) | | | |

## Result

- Raw JSON from `benchmarks/run.py`:
- Honest conclusion (state a loss plainly if there is one):

## Reproduction

```bash
<exact command(s) used, including any env vars set>
```
