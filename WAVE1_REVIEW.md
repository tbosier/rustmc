# RustMC Wave 1 — Engineering Review

**Date:** 2026-07-27
**Reviewer:** team lead (integration worktree)
**Scope:** Tasks 1–5 of the RustMC production roadmap, executed as five isolated worktrees branched from `main` @ `8687933`.

Everything below is either **[verified]** — I ran it myself in the integration worktree — or **[reported]** — claimed by the implementing worktree and not independently re-run. That distinction is kept deliberately; several reported claims turned out to need correction.

---

## 1. Status at a glance

| Task | Scope | PR | State |
|---|---|---|---|
| 1 | Remove silent parameter fallbacks | [#19](https://github.com/tbosier/rustmc/pull/19) | ✅ Approved — verified |
| 2 | Repair & standardize benchmarks | [#18](https://github.com/tbosier/rustmc/pull/18) | ✅ Approved — verified |
| 3 | Statistical validation suite | [#20](https://github.com/tbosier/rustmc/pull/20) | ✅ Approved — spot-verified |
| 4 | Packaging & Python integration | [#17](https://github.com/tbosier/rustmc/pull/17) | ✅ Approved — verified |
| 5 | Compile-once/bind-many **design only** | [#16](https://github.com/tbosier/rustmc/pull/16) | ⏸ Design complete, **4 decisions pending** |

All five are **draft PRs**. Nothing has been merged. No branch was pushed to `main`.

**Baseline before this wave:** `cargo test --all` = 39 tests. No Python test suite existed at all. `cargo fmt --check` dirty; `cargo clippy -- -D warnings` = 27 pre-existing errors.

**After this wave (per-branch, unmerged):** 88 Rust tests, ~254s; 18 Python tests + 5 xfail.

---

## 2. The headline: four real defects, three of them silent

The most important outcome of Wave 1 is not the infrastructure — it is that **rustmc was returning wrong numbers in ways no user could have detected.**

### 2.1 Constrained-parameter coefficients were silently exponentiated — CRITICAL **[verified]**

`build_mu_expr` resolved parameters via `Graph::node_by_name`, which returns the **unconstrained raw node**, rather than through `value_node_map`, which holds the post-transform constrained value.

Any model using a constrained-support prior (**HalfNormal, Exponential, LogNormal, Gamma, Beta**) as a regression coefficient or intercept silently used the wrong node. Measured against the shipped 0.8.0 build: a true coefficient of **2.5 was recovered as 12.14** (≈ exp(2.496)). After the fix: **2.50**.

No error, no warning, plausible-looking posteriors. Anyone who shipped such a model got exponentiated results and had no way to know. Fixed in #19.

### 2.2 Hierarchical models did not work at all through the Python DSL **[verified]**

On `main`, any hierarchical model fails at `sample()` with `ValueError: Unknown param: mu_0`. `examples/hierarchical_example.py` is broken as shipped. This is the feature the README and project memory both present as supported.

**Cross-PR finding:** Task 3 diagnosed this and reported it as an open blocker, having tested against `main`. It is **already fixed by Task 1's PR #19** as a side effect of the same `build_mu_expr` root cause. I verified this directly:

```
# main:            ValueError: Unknown param: mu_0
# PR #19:          sample() OK
#   mu_global 2.541 (true 2.50)   mu_0 2.625 (true 2.651)
```

No separate fix is needed. Task 3's `VALIDATION_RESULTS.md` should be amended before merge so it doesn't record a fixed defect as open.

### 2.3 Divergence counts include warmup — explains the entire benchmark story **[reported, corroborated]**

Task 2 found that rustmc reported nonzero divergences in **every** benchmark where both PyMC backends reported exactly zero, on identical model/data/seed:

| Benchmark | rustmc | PyMC | nutpie |
|---|---|---|---|
| 1-param regression | 22 | 0 | 0 |
| 10-param, 100k obs | 67 | 0 | 0 |
| 500-param regression | 7 | 0 | 0 |
| 100-SKU batch | 1317 | — | 0 |

Task 3, working independently, found the cause: **the divergence counter includes warmup transitions** — 21 reported vs 0 real on a conjugate Normal-Normal, where the analytic posterior is known.

These two findings were produced by different worktrees that never communicated, and they fit exactly. Task 3's evidence that the sampler itself is statistically correct (§2.4) means this is a **reporting** bug, not a sampling bug — but it has been making rustmc look broken in every comparison, and it silently inflates the divergence thresholds in `recovery_suite.rs` (some allow up to 130).

### 2.4 The engine is statistically correct; the bindings were not **[reported]**

This is the reassuring half, and it is well-evidenced. Task 3 reports: every gradient matches finite differences; every constraining transform and Jacobian reproduces its closed form; every analytic posterior matches in **both mean and sd** with zero post-warmup divergences; prior-predictive, posterior-predictive and interval coverage all match closed forms. No bias in autodiff, no bias in any log-density, no scale error.

Crucially, the suite was **negative-control tested**: injecting `sigma * 1.15` into `Normal::prior` caused three tests to fail with exact diagnostics, and the change was reverted (`git diff rust_core/src` empty). A permanent always-on control (`sbc_detects_an_injected_bias`, χ²=65.9, p=1e-11) is included. The suite can detect what it claims to detect.

Notably, SBC at 128 replicates did **not** catch that injected bias — exactly as the published power table predicted. Stated limits are real rather than optimistic.

---

## 3. Other findings worth knowing

**Centered hierarchical parameterisation is genuinely miscalibrated [reported].** SBC at 512 reps: centered funnel `tau` gives χ²=51.75, p=1.5e-8; nominal 75% interval covers 65.6%. Non-centered is clean, so this is the parameterisation, not the engine. It matters because `examples/hierarchical_templates.py` ships the **centered** form and the DSL's auto-non-centering is unreachable.

**`sample_prior_predictive` panics** (Rust `PanicException`) on `@`-auto-promoted vector params — `sample_prior_raw` pushes 1 raw value instead of `n` for Normal/HalfNormal/StudentT/Uniform/Gamma/Beta. Confirmed still failing on PR #19's branch. **[verified]**

**`to_arviz()` discards per-draw divergence flags** it claims aren't stored — they are, in `SampleResult::transitions`. **[reported]**

**rustmc loses the 500-parameter benchmark by 3–6× on ESS/s** (the faer `MatVecMul` path). Previously unstated anywhere. **[reported]**

**The "10,000 models in 70 seconds" claim was false.** The script ran `N_SKUS=100`. The figure appeared in `README.md`, `docs/index.md`, `docs/getting-started.md`, and the batch-inference example. Retracted and replaced with the real 100-SKU table. `batch_sample()` was also defaulting to `chains=1` against nutpie's `chains=4` — a quarter of the work — in the headline comparison. **[reported]**

**Version drift:** `Cargo.lock` pinned `0.7.0` while all three manifests said `0.8.0`. Fixed. **[verified]**

**The wheel ships no type information** — no `py.typed`, no `.pyi`. Correctly deferred to Task 13 but pinned with a strict `xfail`.

**Two sampler observations, unresolved [reported]:** NUTS rejects the whole transition on any divergence where Stan keeps the valid-subtree proposal; and the accept statistic fed to dual averaging is `min(Σw, n_leaves)` rather than `Σ min(1,w)` — likely why warmup divergence rates hit 10–20% and adapted step sizes range 0.38–1.06 on an identical 1-parameter model. These are plausible contributors to §2.3 and deserve a dedicated look.

---

## 4. Process note: two claims that did not survive review

Recorded because they calibrate how much weight to put on agent self-reports.

**Task 5's benchmark harness was stubbed.** The committed harness never ran the sampler — `sample_time` was hardcoded to `Duration::ZERO` — yet the doc quoted "0.21% overhead" to two significant figures off that denominator. On challenge, the worktree established the numbers had been real but produced by an earlier revision that was overwritten before commit. It rewrote the harness so all three shapes run in one invocation with `assert_eq!` guards that make a future stub fail loudly, and committed the raw output. **[verified: sampling now executes]**

While fixing it, it self-reported a second problem I had not spotted: the original table compared **serial** setup against a **parallel** batch wall clock — biased toward its own conclusion. Re-measured honestly, the conclusion held (setup is 0.008–0.032% of a serial fit), but a separate claim ("rebuild is 13× a forward pass") was corrected to a measured range of **1.7×–11.4×**.

**The Task 5 performance case was overstated in early reporting.** The *entanglement* half holds — data and structure are genuinely fused in `Graph`. The *throughput* half does not: rebuild-per-dataset is negligible against a full fit. The real case for Task 5 is **memory and capability**, not fit speed:

- Prediction/prior-predictive paths carry 63–92% rebuild overhead
- A shared design matrix across 10k datasets is currently copied 10k times
- 10k × n=2000 datasets ≈ 1.4 GiB held 3–4× over, all resident before the first draw

Wave 2 should promise memory and capability wins, **not** a batch-throughput win. The benchmark plan has been set to a parity target accordingly.

---

## 5. Merge plan

Conflicts are minimal — three files, all trivial:

| File | Branches | Resolution |
|---|---|---|
| `tests/conftest.py` | T1, T4 | Take T4's (owner); re-apply T1's `target/extmod` sys.path shim |
| `Cargo.lock` | T2, T4 | Identical `0.7.0`→`0.8.0` sync; take either |
| `.gitignore` | T2, T4 | Union |

**Recommended order:** #17 (Task 4, establishes the pytest harness other suites depend on) → #19 (Task 1, the correctness fix) → #20 (Task 3, amend the hierarchical defect entry first) → #18 (Task 2) → #16 (Task 5, docs only, after the four decisions below).

**Merging is a stop-and-ask action and has not been performed.** Recommendation only.

---

## 6. What is left to do

### 6.1 Immediate follow-ups created by this wave

1. **Fix the divergence counter** to exclude warmup (§2.3). Highest priority — it corrupts every diagnostic, benchmark and recovery threshold. Re-baseline `recovery_suite.rs` thresholds afterward.
2. **Investigate the two sampler observations** in §3 (whole-transition rejection; dual-averaging accept statistic). Should be one focused worktree with `nuts.rs` ownership, together with item 1.
3. **Fix `sample_prior_predictive`** panicking on `@`-promoted vector params.
4. **Fix `to_arviz()`** to emit the per-draw divergence flags it already has.
5. **Make auto-non-centering reachable**, and switch `examples/hierarchical_templates.py` off the centered parameterisation (§3).
6. **Amend Task 3's `VALIDATION_RESULTS.md`** — its defect 1 is fixed by #19 (§2.2).
7. **Changelog entry** for the three shapes that now raise `ParameterError`, and for posteriors that will move on constrained-coefficient models (§2.1). Users' numbers will change — correctly — and that must be announced.
8. **Investigate the 500-parameter `MatVecMul` slowdown** (3–6× vs PyMC).
9. **The deferred hygiene pass:** `cargo fmt` + the 27 `clippy -D warnings` errors. Deliberately postponed to avoid colliding with five concurrent worktrees; run it after merge, then turn on the CI gates Task 4 left off.

### 6.2 Decisions needed before Wave 2 starts

Task 5's design is complete but blocked on four calls:

| # | Question | Recommendation |
|---|---|---|
| **OQ-3** | Are dimension names structural or per-binding? | **Structural.** Per-binding names make `required_keys()` undefined without a dataset and stop an artifact declaring its own interface. Asymmetric: structural→per-binding is recoverable later; the reverse permanently gives up the compile-time contract. |
| **OQ-4** | Does Task 8 need per-dataset group counts? | **`Fixed` only.** `FromBinding` makes parameter-vector length per-dataset, which breaks `param_count`/`param_names`/mass-matrix as structural properties. Confirm the use cases fit. |
| **OQ-2** | Preserve `batch_sample`'s legacy seed scheme? | **Product call.** Today's `seed + (idx<<32) + chain` is positional, so inserting one dataset changes every downstream fit. That is a real bug for nightly re-fits. Preserve for compatibility (two schemes until 1.0), or break cleanly in 0.9. |
| **OQ-1** | Rename `Graph` → `ModelStructure`? | **No.** Large diff across concurrently-edited files, no functional gain. |

### 6.3 Roadmap remaining

**Milestone 1** (Tasks 1–14, state-space excluded) is roughly **1/3 complete**. Tasks 1–4 are done pending merge; Task 5 is designed but unimplemented.

Still to do: **Task 5 implementation** (staged S1–S7; S1–S3 sequential, S4/S5 parallel after), **Task 6** named dims/coords, **Task 7** expression graph, **Task 8** group indexing, **Task 9** memory-efficient binding, **Task 10** diagnostics & result shapes, **Task 11** ArviZ-native export, **Task 12** automatic inference selection, **Task 13** typed Python API, **Task 14** prediction on new data. Then **Task 15** state-space/Kalman, design-first.

**Sequencing:** Wave 2 cannot fully parallelise. Task 5's S1–S3 rewrite the `Evaluator` signature and remove the data fields from `Graph`; nothing else should be in those files concurrently. Tasks 6–8 must share one shape/indexing design and should be one worktree, starting only after OQ-3 is settled.

**Recommendation: insert a correctness worktree before Wave 2.** Items 1–4 of §6.1 are all sampler/diagnostics defects, they are cheap, and they block honest benchmarking of everything built afterwards. Wave 2 should not start on top of a divergence counter that is known to be wrong.

---

## 7. Assessment

The engine's mathematical core is in better shape than its bindings. Autodiff, the log-densities, the transforms and the analytic posteriors all check out under real scrutiny including negative controls. Every defect found this wave was in the **binding layer** — parameter resolution, predictive sampling, diagnostic reporting — or in the claims made about the engine, not in the mathematics.

That is a good position to be in, because the binding layer is exactly what Tasks 5–14 rebuild.

The honest caveat for the "defacto standard library" ambition: rustmc's supported model surface is currently narrow. Six likelihoods; `ParamRef` hyperparameters on only four priors; expression algebra limited to `param * key`, `param @ key` and `+`; no multivariate normal, no LKJ, no mixtures, no truncation, no missing-data handling; hierarchical × vector explicitly rejected. The roadmap's narrow positioning — *compile one model, fit thousands of datasets* — is the right call, and the wave's results support it. Competing with PyMC on expressiveness is not the winnable fight; being the fastest, most trustworthy engine for repeated structured fits is.
