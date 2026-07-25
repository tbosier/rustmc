//! Numerical stability, extreme-but-valid inputs, determinism, and failure
//! modes.
//!
//! Everything here is either exact (no Monte Carlo tolerance needed) or
//! asserted against a closed form. Where a statistical tolerance is needed the
//! justification is stated inline.

mod common;

use common::{
    assert_converged, assert_mean_within_mcse, assert_post_warmup_divergence_rate,
    assert_sd_within_mcse, diag, linear_gaussian_posterior, nuts, Rng,
};
use rustmc_core::autodiff::{eval_logp, grad_logp, Evaluator};
use rustmc_core::distributions::{HalfNormal, Normal};
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{sample as run_sample, SamplerConfig, SamplerType};

// ── Determinism ────────────────────────────────────────────────────────────

/// The same graph, config and seed must produce bit-identical draws.
///
/// Without this, every other test in the suite is unfalsifiable: a failure
/// could always be blamed on the RNG. `num_threads = 1` is deliberate; the
/// cross-thread case is asserted separately below.
#[test]
fn sampling_is_bit_reproducible_for_a_fixed_seed() {
    let build = || {
        let mut rng = Rng::new(5150);
        let n = 40;
        let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 0.5 * xi + rng.normal()).collect();
        let mut g = Graph::new();
        let a = Normal::prior(&mut g, "a", 0.0, 3.0);
        let b = Normal::prior(&mut g, "b", 0.0, 3.0);
        let s = HalfNormal::prior(&mut g, "sigma", 1.0);
        let d0 = g.store_data_vec(vec![1.0; n]);
        let d1 = g.store_data_vec(x);
        let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
        let obs = g.add_obs_data(y);
        g.normal_obs_logp(mu, s, obs);
        g
    };

    let first = nuts(build(), 777, 2, 120, 120);
    let second = nuts(build(), 777, 2, 120, 120);
    assert_eq!(
        first.samples, second.samples,
        "draws differ across identical runs"
    );
    assert_eq!(first.step_sizes, second.step_sizes, "step sizes differ");
    assert_eq!(
        first.divergences, second.divergences,
        "divergence counts differ"
    );
}

/// Chain results must not depend on the size of the Rayon pool: each chain is
/// seeded from `seed + chain_index`, so thread scheduling must be irrelevant.
#[test]
fn chain_results_are_independent_of_thread_count() {
    let build = || {
        let mut g = Graph::new();
        let mu = Normal::prior(&mut g, "mu", 0.0, 2.0);
        let mu_vec = g.scalar_broadcast(mu);
        let s = g.add_constant(1.0);
        let obs = g.add_obs_data(vec![0.4, -0.2, 1.1, 0.7, -0.9]);
        g.normal_obs_logp(mu_vec, s, obs);
        g
    };
    let run = |threads: usize| {
        run_sample(
            build(),
            SamplerConfig {
                sampler: SamplerType::Nuts,
                num_chains: 4,
                num_draws: 100,
                num_warmup: 100,
                step_size: 0.0,
                num_leapfrog_steps: 15,
                max_tree_depth: 10,
                seed: 4242,
                num_threads: threads,
                show_progress: false,
            },
        )
        .expect("sampling failed")
    };
    assert_eq!(
        run(1).samples,
        run(4).samples,
        "draws depend on the thread pool size"
    );
}

// ── Extreme but valid inputs ───────────────────────────────────────────────

/// Observation noise six orders of magnitude apart on either side of 1.
///
/// The posterior is exactly Gaussian in both cases, so both the location and
/// the scale of the answer are checked, not merely that nothing crashed.
#[test]
fn extreme_observation_scales_still_give_the_analytic_posterior() {
    for &sigma in &[1e-4_f64, 1e4] {
        let mut rng = Rng::new(31337);
        let n = 30;
        let mu_true = 2.0;
        let s0 = 10.0 * sigma.max(1.0);
        let y: Vec<f64> = (0..n).map(|_| rng.normal_with(mu_true, sigma)).collect();

        let prec = 1.0 / (s0 * s0) + n as f64 / (sigma * sigma);
        let post_mean = (y.iter().sum::<f64>() / (sigma * sigma)) / prec;
        let post_sd = (1.0 / prec).sqrt();

        let mut g = Graph::new();
        let mu = Normal::prior(&mut g, "mu", 0.0, s0);
        let mu_vec = g.scalar_broadcast(mu);
        let sn = g.add_constant(sigma);
        let obs = g.add_obs_data(y);
        g.normal_obs_logp(mu_vec, sn, obs);

        let result = nuts(g, 50_001, 4, 600, 600);
        let report = result.diagnostics();
        assert_converged(&report, 1.03);
        assert_post_warmup_divergence_rate(&result, 0.01);
        assert_mean_within_mcse(&report, "mu", post_mean, 300.0, 0.0);
        assert_sd_within_mcse(&report, "mu", post_sd, 300.0, 0.0);
    }
}

/// A single observation. Everything downstream of `n_obs` must handle `n = 1`
/// (SIMD-style loops, variance accumulators, matrix row counts).
#[test]
fn a_single_observation_still_gives_the_analytic_posterior() {
    let sigma = 1.3_f64;
    let s0 = 2.0_f64;
    let y = 0.75_f64;
    let prec = 1.0 / (s0 * s0) + 1.0 / (sigma * sigma);
    let post_mean = (y / (sigma * sigma)) / prec;
    let post_sd = (1.0 / prec).sqrt();

    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, s0);
    let mu_vec = g.scalar_broadcast(mu);
    let sn = g.add_constant(sigma);
    let obs = g.add_obs_data(vec![y]);
    g.normal_obs_logp(mu_vec, sn, obs);

    let result = nuts(g, 50_002, 4, 800, 500);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergences_zero(&result);
    assert_mean_within_mcse(&report, "mu", post_mean, 400.0, 0.0);
    assert_sd_within_mcse(&report, "mu", post_sd, 400.0, 0.0);
}

fn assert_post_warmup_divergences_zero(result: &rustmc_core::sampler::SampleResult) {
    common::assert_post_warmup_divergences(result, 0);
}

/// A large observation vector through the fused path. Catches accumulation
/// error and any `n`-dependent indexing bug that only bites past a chunk size.
#[test]
fn large_observation_vector_gives_the_analytic_posterior() {
    let mut rng = Rng::new(90210);
    let n = 6_000;
    let sigma = 1.0;
    let beta_true = [0.5, -0.25];
    let prior_sd = [5.0, 5.0];

    let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![1.0, x[i]]).collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| beta_true[0] * r[0] + beta_true[1] * r[1] + rng.normal_with(0.0, sigma))
        .collect();
    let exact = linear_gaussian_posterior(&rows, &y, sigma, &[0.0, 0.0], &prior_sd);

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, prior_sd[0]);
    let b = Normal::prior(&mut g, "b", 0.0, prior_sd[1]);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let sn = g.add_constant(sigma);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sn, obs);

    let result = nuts(g, 50_003, 2, 250, 250);
    let report = result.diagnostics();
    assert_converged(&report, 1.05);
    for (k, name) in ["a", "b"].iter().enumerate() {
        assert_mean_within_mcse(&report, name, exact.mean[k], 100.0, 0.0);
        assert_sd_within_mcse(&report, name, exact.sd(k), 100.0, 0.0);
    }
}

/// Perfectly collinear predictors: the likelihood is flat along one direction,
/// so the posterior is exactly the prior in that direction. A sampler that
/// blew up (NaNs, infinite step sizes, unbounded drift) would fail here.
#[test]
fn perfectly_collinear_predictors_fall_back_to_the_prior_direction() {
    let mut rng = Rng::new(1618);
    let n = 60;
    let sigma = 0.5;
    let prior_sd = 1.0;
    let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    // x2 is an exact copy of x1, so only (b1 + b2) is identified.
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![x[i], x[i]]).collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| 0.6 * r[0] + rng.normal_with(0.0, sigma))
        .collect();
    let exact = linear_gaussian_posterior(&rows, &y, sigma, &[0.0, 0.0], &[prior_sd, prior_sd]);

    let mut g = Graph::new();
    let b1 = Normal::prior(&mut g, "b1", 0.0, prior_sd);
    let b2 = Normal::prior(&mut g, "b2", 0.0, prior_sd);
    let d1 = g.store_data_vec(x.clone());
    let d2 = g.store_data_vec(x);
    let mu = g.fused_linear_mu(vec![b1, b2], vec![d1, d2], None);
    let sn = g.add_constant(sigma);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sn, obs);

    let result = nuts(g, 50_004, 4, 1000, 1000);
    let report = result.diagnostics();
    for p in &report.params {
        assert!(
            p.mean.is_finite() && p.std.is_finite(),
            "{}: non-finite summary",
            p.name
        );
    }
    assert_converged(&report, 1.10);
    // ESS along the unidentified direction is low by construction, so 100 is
    // the gate; the posterior is still exactly Gaussian and both marginals
    // are asserted against it.
    for (k, name) in ["b1", "b2"].iter().enumerate() {
        assert_mean_within_mcse(&report, name, exact.mean[k], 100.0, 0.0);
        assert_sd_within_mcse(&report, name, exact.sd(k), 100.0, 0.0);
    }
}

/// Data values large enough that a naive `sum(x^2)` would lose precision.
#[test]
fn large_magnitude_predictors_do_not_destroy_the_gradient() {
    let n = 50;
    let mut rng = Rng::new(2024);
    let x: Vec<f64> = (0..n).map(|_| 1e6 * rng.normal()).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1e-6 * xi + rng.normal()).collect();

    let mut g = Graph::new();
    let b = Normal::prior(&mut g, "b", 0.0, 1.0);
    let d1 = g.store_data_vec(x);
    let mu = g.fused_linear_mu(vec![b], vec![d1], None);
    let sn = g.add_constant(1.0);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sn, obs);

    // Analytic check of the gradient at a point, independent of the sampler.
    for &p in &[-1e-6, 0.0, 1e-6, 5e-6] {
        let (logp, grad) = grad_logp(&g, &[p]);
        assert!(logp.is_finite(), "logp not finite at b = {p}");
        let h = 1e-11;
        let numeric = (eval_logp(&g, &[p + h]) - eval_logp(&g, &[p - h])) / (2.0 * h);
        let tol = 1e-4 * numeric.abs().max(grad[0].abs()) + 1e-3;
        assert!(
            (grad[0] - numeric).abs() <= tol,
            "gradient {} vs finite difference {} at b = {p}",
            grad[0],
            numeric
        );
    }
}

// ── Failure modes ──────────────────────────────────────────────────────────

#[test]
fn mismatched_data_lengths_are_rejected_before_sampling() {
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 1.0);
    let d0 = g.store_data_vec(vec![1.0; 10]);
    let mu = g.fused_linear_mu(vec![a], vec![d0], None);
    let sn = g.add_constant(1.0);
    let obs = g.add_obs_data(vec![0.0; 7]); // wrong length
    g.normal_obs_logp(mu, sn, obs);

    let err = g
        .validate_shapes()
        .expect_err("mismatched lengths must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("length") && msg.contains("expected"),
        "shape error message is not actionable: {msg}"
    );

    let mut g2 = Graph::new();
    let a = Normal::prior(&mut g2, "a", 0.0, 1.0);
    let d0 = g2.store_data_vec(vec![1.0; 10]);
    let mu = g2.fused_linear_mu(vec![a], vec![d0], None);
    let sn = g2.add_constant(1.0);
    let obs = g2.add_obs_data(vec![0.0; 7]);
    g2.normal_obs_logp(mu, sn, obs);
    let result = run_sample(
        g2,
        SamplerConfig {
            show_progress: false,
            num_draws: 5,
            num_warmup: 5,
            num_chains: 1,
            ..Default::default()
        },
    );
    assert!(
        result.is_err(),
        "sample() accepted a graph with inconsistent vector lengths"
    );
}

#[test]
fn matrix_payload_shape_mismatch_is_rejected() {
    let mut g = Graph::new();
    let s = g.add_vector_params("beta", 3);
    g.vector_normal_logp(s, 3, 0.0, 1.0);
    // Claim 10x3 but supply only 20 values.
    let m = g.store_matrix(vec![0.0; 20], 10, 3);
    let mu = g.mat_vec_mul(m, s, 3, None);
    let sn = g.add_constant(1.0);
    let obs = g.add_obs_data(vec![0.0; 10]);
    g.normal_obs_logp(mu, sn, obs);

    let err = g
        .validate_shapes()
        .expect_err("bad matrix payload must be rejected");
    assert!(
        err.to_string().contains("shape"),
        "matrix shape error is not actionable: {err}"
    );
}

#[test]
fn matrix_row_count_must_match_the_observation_vector() {
    let mut g = Graph::new();
    let s = g.add_vector_params("beta", 2);
    g.vector_normal_logp(s, 2, 0.0, 1.0);
    let m = g.store_matrix(vec![0.0; 12], 6, 2);
    let mu = g.mat_vec_mul(m, s, 2, None);
    let sn = g.add_constant(1.0);
    let obs = g.add_obs_data(vec![0.0; 9]);
    g.normal_obs_logp(mu, sn, obs);

    assert!(
        g.validate_shapes().is_err(),
        "matrix row count / observation length mismatch was not caught"
    );
}

#[test]
fn evaluator_construction_reports_shape_errors_rather_than_panicking() {
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 1.0);
    let d0 = g.store_data_vec(vec![1.0; 5]);
    let mu = g.fused_linear_mu(vec![a], vec![d0], None);
    let sn = g.add_constant(1.0);
    let obs = g.add_obs_data(vec![0.0; 4]);
    g.normal_obs_logp(mu, sn, obs);

    assert!(
        Evaluator::try_new(&g).is_err(),
        "Evaluator::try_new accepted an inconsistent graph"
    );
}

/// A model with no observations at all is legal (it is a prior sample) and
/// must not be mistaken for a shape error.
#[test]
fn prior_only_graph_is_valid_and_samples() {
    let mut g = Graph::new();
    Normal::prior(&mut g, "x", 1.0, 2.0);
    assert_eq!(
        g.validate_shapes().expect("prior-only graph must validate"),
        0
    );
    let result = nuts(g, 50_005, 2, 200, 200);
    let report = result.diagnostics();
    assert!(diag(&report, "x").mean.is_finite());
}
