//! Simulation-Based Calibration (Talts, Betancourt, Simpson, Vehtari & Gelman,
//! 2018, arXiv:1804.06788).
//!
//! # What this proves
//!
//! For any correct Bayesian sampler, if you draw `theta~ ~ p(theta)`, simulate
//! `y ~ p(y | theta~)`, and then draw `L` independent samples from
//! `p(theta | y)`, the rank of `theta~` among those `L` draws is uniform on
//! `{0, ..., L}`. This holds for *every* model and *every* parameter, and it
//! fails whenever the sampled distribution differs from the intended posterior
//! in any way — a shifted mean, a compressed variance, a wrong Jacobian, a
//! non-reversible transition kernel. It is the single most sensitive test in
//! this suite and the one most likely to expose a subtle bias.
//!
//! # Configuration and why
//!
//! * `L = 63` posterior draws per replicate, kept after thinning by `THIN = 8`
//!   from a chain of `63 * 8 = 504` post-warmup draws. Thinning is what makes
//!   the "L independent draws" premise approximately true; without it,
//!   autocorrelation inflates the middle of the rank histogram and produces
//!   false alarms. `L + 1 = 64` possible ranks bin exactly into 8 bins of 8.
//!
//! * Ranks are binned into `BINS = 8` cells and tested with a chi-square
//!   goodness-of-fit statistic against uniformity (`df = 7`). We fail below
//!   `P_THRESHOLD = 1e-3`. With ~14 chi-square tests across this file the
//!   family-wise false-alarm probability is ~1.4%; each individual test is
//!   therefore effectively never flaky, while a systematic bias — which grows
//!   the statistic non-centrally in the number of replicates — is caught.
//!
//! * The fast test uses `FAST_REPS = 128` replicates; the opt-in test uses
//!   `FULL_REPS = 512`.
//!
//! # Power (what these tests can and cannot detect)
//!
//! The non-centrality of the chi-square statistic is
//! `N * sum_k (p_k - 1/8)^2 / (1/8)`, where `p_k` is the rank-bin probability
//! under the perturbed posterior. Rejection at `p < 1e-3` needs `chi2 > 24.3`.
//! Solving for the perturbation size gives the detection thresholds:
//!
//! | perturbation                        | detected at 128 reps | at 512 reps |
//! |-------------------------------------|----------------------|-------------|
//! | posterior mean shifted by `d` sd    | `d >= 0.45`          | `d >= 0.22` |
//! | posterior sd inflated by factor `c` | `c >= 1.5`           | `c >= 1.22` |
//!
//! So these tests rule out gross bias, not a 5% scale error. That limit is a
//! property of SBC replicate counts, not of the assertion, and raising
//! `FULL_REPS` is the only way to tighten it. `sbc_detects_an_injected_bias`
//! is a live negative control that fails if the machinery ever goes inert.
//!
//! # Running the full suite
//!
//! ```text
//! cargo test -p rustmc_core --test sbc --release -- --ignored --nocapture
//! ```
//!
//! It prints every rank histogram, chi-square statistic and p-value, so a
//! failure is diagnosable rather than just red.
//!
//! # Posterior credible-interval calibration
//!
//! `run_sbc` also asserts that the empirical coverage of central posterior
//! credible intervals matches their nominal level. This is `VALIDATION.md`
//! workstream 2 ("predictive intervals contain observed values at the expected
//! rate") applied at the parameter level, where the nominal rate is exact
//! rather than simulated. It is computed from the same ranks at no extra cost.

mod common;

use common::{chi_square_sf, chi_square_uniform, nuts, Rng};
use rustmc_core::distributions::{HalfNormal, Normal};
use rustmc_core::graph::Graph;

const L: usize = 63;
const THIN: usize = 8;
const WARMUP: usize = 400;
const BINS: usize = 8;
const P_THRESHOLD: f64 = 1e-3;
const FAST_REPS: usize = 128;
const FULL_REPS: usize = 512;

/// One SBC replicate: a graph, the true parameter values in *constrained*
/// (reported) space, and the parameter names to rank.
struct Replicate {
    graph: Graph,
    truth: Vec<f64>,
    names: Vec<String>,
}

/// Per-parameter SBC outcome.
struct SbcOutcome {
    name: String,
    chi2: f64,
    df: usize,
    p_value: f64,
    hist: Vec<usize>,
    /// (nominal, empirical, binomial standard error) for each interval tested.
    coverage: Vec<(f64, f64, f64)>,
}

/// Central credible intervals checked as a by-product of the ranks: the
/// fraction of replicates whose rank lands in the central `b - a + 1` of the
/// `L + 1` possible ranks is exactly the empirical coverage of the
/// corresponding central posterior interval.
const COVERAGE_WINDOWS: [(usize, usize); 2] = [(8, 55), (3, 60)];

/// Run SBC for a model and return one outcome per ranked parameter.
fn sbc_report<F>(label: &str, reps: usize, mut build: F) -> Vec<SbcOutcome>
where
    F: FnMut(&mut Rng) -> Replicate,
{
    let mut rng = Rng::new(0x5BC_0000 + label.len() as u64);
    let mut histograms: Vec<Vec<usize>> = Vec::new();
    let mut all_ranks: Vec<Vec<usize>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    for rep in 0..reps {
        let Replicate {
            graph,
            truth,
            names: pnames,
        } = build(&mut rng);
        if histograms.is_empty() {
            histograms = vec![vec![0usize; BINS]; pnames.len()];
            all_ranks = vec![Vec::with_capacity(reps); pnames.len()];
            names = pnames.clone();
        }

        // One chain; ranks must come from a single exchangeable set of draws.
        let result = nuts(graph, 900_000 + rep as u64 * 7919, 1, L * THIN, WARMUP);
        let chain = &result.samples[0];
        let param_names = &result.param_names;

        for (pi, pname) in pnames.iter().enumerate() {
            let idx = param_names
                .iter()
                .position(|n| n == pname)
                .unwrap_or_else(|| panic!("{label}: no parameter named {pname}"));
            let draws: Vec<f64> = (0..L).map(|k| chain[k * THIN][idx]).collect();
            let rank = draws.iter().filter(|&&v| v < truth[pi]).count();
            let bin = rank * BINS / (L + 1);
            histograms[pi][bin.min(BINS - 1)] += 1;
            all_ranks[pi].push(rank);
        }
    }

    let mut out = Vec::new();
    for (pi, name) in names.iter().enumerate() {
        let (chi2, df) = chi_square_uniform(&histograms[pi]);
        let p_value = chi_square_sf(chi2, df);
        let coverage = COVERAGE_WINDOWS
            .iter()
            .map(|&(a, b)| {
                let q = (b - a + 1) as f64 / (L + 1) as f64;
                let hits = all_ranks[pi].iter().filter(|&&r| r >= a && r <= b).count();
                let emp = hits as f64 / reps as f64;
                let se = (q * (1.0 - q) / reps as f64).sqrt();
                (q, emp, se)
            })
            .collect();
        println!(
            "SBC {label:<34} {name:<10} reps={reps:<4} chi2={chi2:7.2} df={df} p={p_value:.4}  \
hist={:?}  coverage={:?}",
            histograms[pi],
            COVERAGE_WINDOWS
                .iter()
                .map(|&(a, b)| {
                    let q = (b - a + 1) as f64 / (L + 1) as f64;
                    let hits = all_ranks[pi].iter().filter(|&&r| r >= a && r <= b).count();
                    (
                        (100.0 * q).round() / 100.0,
                        (1000.0 * hits as f64 / reps as f64).round() / 1000.0,
                    )
                })
                .collect::<Vec<_>>()
        );
        out.push(SbcOutcome {
            name: name.clone(),
            chi2,
            df,
            p_value,
            hist: histograms[pi].clone(),
            coverage,
        });
    }
    out
}

/// Run SBC and assert every rank histogram is uniform and every credible
/// interval attains its nominal coverage.
///
/// Coverage tolerance: 3.5 binomial standard errors, a ~0.05% per-assertion
/// false-alarm rate.
fn run_sbc<F>(label: &str, reps: usize, build: F)
where
    F: FnMut(&mut Rng) -> Replicate,
{
    let outcomes = sbc_report(label, reps, build);
    let mut failures = Vec::new();
    for o in &outcomes {
        for &(q, emp, se) in &o.coverage {
            if (emp - q).abs() > 3.5 * se {
                failures.push(format!(
                    "{}: central {:.1}% credible interval covered {:.1}% over {reps} \
                     replicates (|diff| {:.4} > 3.5 * binomial se {:.4})",
                    o.name,
                    100.0 * q,
                    100.0 * emp,
                    (emp - q).abs(),
                    3.5 * se
                ));
            }
        }
        if o.p_value < P_THRESHOLD {
            failures.push(format!(
                "{}: rank histogram not uniform, chi2={:.2} df={} p={:.2e} hist={:?}",
                o.name, o.chi2, o.df, o.p_value, o.hist
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: SBC calibration failed:\n  {}",
        failures.join("\n  ")
    );
}

// ── Models ─────────────────────────────────────────────────────────────────

/// Location only: `mu ~ N(0, 1)`, `y_i ~ N(mu, 1)`.
/// Targets the core Normal density and the `ScalarBroadcast` + observation path.
fn location_model(rng: &mut Rng) -> Replicate {
    let n = 8;
    let mu_true = rng.normal();
    let y: Vec<f64> = (0..n).map(|_| rng.normal_with(mu_true, 1.0)).collect();

    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 1.0);
    let mu_vec = g.scalar_broadcast(mu);
    let s = g.add_constant(1.0);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu_vec, s, obs);

    Replicate {
        graph: g,
        truth: vec![mu_true],
        names: vec!["mu".into()],
    }
}

/// Scale only: `sigma ~ HalfNormal(1)`, `y_i ~ N(0, sigma)`.
///
/// This is the highest-value SBC model in the file: it is the only one whose
/// correctness depends on the exp transform's Jacobian being right, and a
/// wrong Jacobian produces a *perfectly converged* chain with a systematically
/// shifted rank histogram.
fn scale_model(rng: &mut Rng) -> Replicate {
    let n = 8;
    // HalfNormal(1) draw.
    let sigma_true = rng.normal().abs().max(1e-6);
    let y: Vec<f64> = (0..n).map(|_| rng.normal_with(0.0, sigma_true)).collect();

    let mut g = Graph::new();
    let sigma = HalfNormal::prior(&mut g, "sigma", 1.0);
    let zero = g.add_constant(0.0);
    let mu_vec = g.scalar_broadcast(zero);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu_vec, sigma, obs);

    Replicate {
        graph: g,
        truth: vec![sigma_true],
        names: vec!["sigma".into()],
    }
}

/// Regression with a learned scale: `a, b ~ N(0, 1)`, `sigma ~ HalfNormal(1)`,
/// `y_i ~ N(a + b x_i, sigma)`. Targets `FusedLinearMu` and the joint geometry
/// of location and scale parameters.
fn regression_model(rng: &mut Rng) -> Replicate {
    let n = 12;
    let a_true = rng.normal();
    let b_true = rng.normal();
    let sigma_true = rng.normal().abs().max(1e-6);
    let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| rng.normal_with(a_true + b_true * xi, sigma_true))
        .collect();

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 1.0);
    let b = Normal::prior(&mut g, "b", 0.0, 1.0);
    let sigma = HalfNormal::prior(&mut g, "sigma", 1.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sigma, obs);

    Replicate {
        graph: g,
        truth: vec![a_true, b_true, sigma_true],
        names: vec!["a".into(), "b".into(), "sigma".into()],
    }
}

/// Non-centered two-level hierarchy: `mu ~ N(0, 1)`, `tau ~ HalfNormal(1)`,
/// `theta_j = mu + tau z_j` with `z_j ~ N(0, 1)`, `y_j ~ N(theta_j, 1)`.
/// Targets the hierarchical wiring the Python DSL auto-non-centers onto.
fn hierarchical_model(rng: &mut Rng) -> Replicate {
    let j = 5;
    let mu_true = rng.normal();
    let tau_true = rng.normal().abs().max(1e-6);
    let z_true: Vec<f64> = (0..j).map(|_| rng.normal()).collect();

    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 1.0);
    let tau = HalfNormal::prior(&mut g, "tau", 1.0);
    for (jj, &z_t) in z_true.iter().enumerate() {
        let y_j = rng.normal_with(mu_true + tau_true * z_t, 1.0);
        let z = Normal::prior(&mut g, &format!("z_{jj}"), 0.0, 1.0);
        let tz = g.mul(tau, z);
        let theta = g.add(mu, tz);
        let yn = g.add_constant(y_j);
        let sn = g.add_constant(1.0);
        g.normal_logp(yn, theta, sn);
    }

    Replicate {
        graph: g,
        truth: vec![mu_true, tau_true, z_true[0]],
        names: vec!["mu".into(), "tau".into(), "z_0".into()],
    }
}

/// **Centered** two-level hierarchy: `theta_j ~ N(mu, tau)` directly, with
/// `tau ~ HalfNormal(1)`. This is the classic funnel geometry and it produces
/// divergent transitions at any practical step size.
///
/// It is here deliberately. `nuts::run_chain` rejects the *entire* transition
/// whenever any subtree diverges (`if !tree_stats.diverging { current = ... }`)
/// rather than keeping the proposal drawn from the valid part of the
/// trajectory, which is what Stan does. That modification is not obviously
/// reversible, so it is a plausible source of bias — and this is the model
/// where such a bias would show up. See `docs`/the validation report.
fn centered_hierarchical_model(rng: &mut Rng) -> Replicate {
    let j = 5;
    let mu_true = rng.normal();
    let tau_true = rng.normal().abs().max(1e-6);
    let theta_true: Vec<f64> = (0..j).map(|_| rng.normal_with(mu_true, tau_true)).collect();

    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 1.0);
    let tau = HalfNormal::prior(&mut g, "tau", 1.0);
    for (jj, &t_t) in theta_true.iter().enumerate() {
        let y_j = rng.normal_with(t_t, 1.0);
        let theta = Normal::prior_with_nodes(&mut g, &format!("theta_{jj}"), mu, tau);
        let yn = g.add_constant(y_j);
        let sn = g.add_constant(1.0);
        g.normal_logp(yn, theta, sn);
    }

    Replicate {
        graph: g,
        truth: vec![mu_true, tau_true, theta_true[0]],
        names: vec!["mu".into(), "tau".into(), "theta_0".into()],
    }
}

/// Matrix-vector regression: `beta ~ N(0, 1)^p`, `y ~ N(X beta, 1)`.
/// Targets the faer `MatVecMul` gradient and the `VectorNormalLogP` prior.
fn matvec_model(rng: &mut Rng) -> Replicate {
    let n = 15;
    let p = 3;
    let beta_true: Vec<f64> = (0..p).map(|_| rng.normal()).collect();
    let mut flat = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..p).map(|_| rng.normal()).collect();
        let mu: f64 = row.iter().zip(&beta_true).map(|(a, b)| a * b).sum();
        y.push(rng.normal_with(mu, 1.0));
        flat.extend(row);
    }

    let mut g = Graph::new();
    let s = g.add_vector_params("beta", p);
    g.vector_normal_logp(s, p, 0.0, 1.0);
    let m = g.store_matrix(flat, n, p);
    let mu = g.mat_vec_mul(m, s, p, None);
    let sig = g.add_constant(1.0);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sig, obs);

    Replicate {
        graph: g,
        truth: vec![beta_true[0], beta_true[1]],
        names: vec!["beta[0]".into(), "beta[1]".into()],
    }
}

// ── Fast (always-on) tests ─────────────────────────────────────────────────

#[test]
fn sbc_location_model_is_calibrated() {
    run_sbc("location N(0,1) / N(mu,1)", FAST_REPS, location_model);
}

#[test]
fn sbc_scale_model_is_calibrated() {
    run_sbc("scale HalfNormal(1) / N(0,sigma)", FAST_REPS, scale_model);
}

// ── Opt-in (slower, higher power) tests ────────────────────────────────────

#[test]
#[ignore = "full SBC run; opt in with --ignored"]
fn sbc_location_model_is_calibrated_full() {
    run_sbc("location (full)", FULL_REPS, location_model);
}

#[test]
#[ignore = "full SBC run; opt in with --ignored"]
fn sbc_scale_model_is_calibrated_full() {
    run_sbc("scale (full)", FULL_REPS, scale_model);
}

#[test]
#[ignore = "full SBC run; opt in with --ignored"]
fn sbc_regression_model_is_calibrated_full() {
    run_sbc(
        "regression a+bx, learned sigma",
        FULL_REPS,
        regression_model,
    );
}

#[test]
#[ignore = "full SBC run; opt in with --ignored"]
fn sbc_hierarchical_model_is_calibrated_full() {
    run_sbc("hierarchical non-centered", FULL_REPS, hierarchical_model);
}

#[test]
#[ignore = "full SBC run; opt in with --ignored"]
fn sbc_matvec_model_is_calibrated_full() {
    run_sbc("matvec X @ beta", FULL_REPS, matvec_model);
}

/// Characterisation test for a **reproduced miscalibration**.
///
/// The centered parameterisation of the two-level hierarchy is the classic
/// Neal funnel: NUTS cannot reach the neck at small `tau`, so `tau`'s
/// posterior is systematically truncated from below and its credible
/// intervals under-cover. This is measured, not assumed:
///
/// ```text
/// tau: central 75% credible interval covered 65.6% over 512 replicates
///      (4.9 binomial standard errors low)
/// ```
///
/// The non-centered version of the same model (`sbc_hierarchical_model_*`) is
/// calibrated, which is what localises this to the parameterisation rather
/// than to the engine. It matters because
/// `examples/hierarchical_templates.py` builds the *centered* form and the
/// Python DSL's auto-non-centering is unreachable (see the pinned defect in
/// `tests/test_statistical_engine_bugs.py`).
///
/// This test asserts the miscalibration is still present. If it starts
/// passing — because the geometry handling improved — that is good news and
/// the test (and the docs it points at) should be updated.
#[test]
#[ignore = "full SBC run; opt in with --ignored"]
fn sbc_centered_hierarchical_model_is_known_miscalibrated() {
    let outcomes = sbc_report(
        "hierarchical CENTERED (funnel)",
        FULL_REPS,
        centered_hierarchical_model,
    );
    let tau = outcomes
        .iter()
        .find(|o| o.name == "tau")
        .expect("tau outcome");
    let (q, emp, se) = tau.coverage[0];
    assert!(
        emp < q - 3.0 * se,
        "the centered funnel's tau credible interval is no longer under-covering \
         (nominal {:.1}%, empirical {:.1}%, se {:.4}); update this test and the \
         hierarchical docs/examples",
        100.0 * q,
        100.0 * emp,
        se
    );
}

/// Negative control: rank the true value against posterior draws that have
/// been deliberately shifted by 0.7 posterior sd. SBC must reject this.
///
/// Guards against the SBC machinery itself being inert — a wrong rank formula,
/// an off-by-one in the binning, a chi-square implementation that never fires.
/// At 128 replicates a 0.7 sd shift gives an expected chi-square of ~64
/// (p ~ 2e-11), so this control has a large margin and is not itself flaky.
#[test]
fn sbc_detects_an_injected_bias() {
    let reps = FAST_REPS;
    let mut rng = Rng::new(0xDEADBEEF);
    let mut hist = vec![0usize; BINS];

    for rep in 0..reps {
        let Replicate { graph, truth, .. } = location_model(&mut rng);
        let result = nuts(graph, 900_000 + rep as u64 * 7919, 1, L * THIN, WARMUP);
        let chain = &result.samples[0];
        let draws: Vec<f64> = (0..L).map(|k| chain[k * THIN][0]).collect();
        let mean = draws.iter().sum::<f64>() / L as f64;
        let sd = (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (L as f64 - 1.0)).sqrt();
        // Inject a 0.7-sd location bias into the posterior draws.
        let perturbed: Vec<f64> = draws.iter().map(|v| v + 0.7 * sd).collect();
        let rank = perturbed.iter().filter(|&&v| v < truth[0]).count();
        hist[(rank * BINS / (L + 1)).min(BINS - 1)] += 1;
    }

    let (stat, df) = chi_square_uniform(&hist);
    let p = chi_square_sf(stat, df);
    println!(
        "SBC negative control (+0.7 sd shift): chi2={stat:.2} df={df} p={p:.3e} hist={hist:?}"
    );
    assert!(
        p < P_THRESHOLD,
        "SBC failed to detect a 0.7-sd posterior location bias \
         (chi2={stat:.2}, p={p:.3e}, hist={hist:?}); the calibration test has no power"
    );
}
