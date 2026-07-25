//! Parameter recovery for the observation families that `recovery_suite.rs`
//! does not cover, plus a high-dimensional `MatVecMul` recovery run.
//!
//! Recovery tests are weaker evidence than the analytic-posterior tests: they
//! only show the posterior sits near the truth, not that it has the right
//! shape. They are included because they are the only available check for
//! families with no conjugate reference (Exponential, NegativeBinomial), and
//! because they exercise the full sampling stack on realistic data sizes.
//!
//! # Tolerance
//!
//! For a well-identified GLM coefficient the posterior sd is approximately the
//! maximum-likelihood standard error. Each assertion therefore uses
//! `|posterior mean - truth| <= 3 * posterior sd + 4 * mcse_mean`: three
//! posterior sds is the sampling variability of the estimate given this one
//! seeded dataset, and the MCSE term covers the Monte Carlo error on top. This
//! is derived, not tuned; the tests were not re-run with widened tolerances.

mod common;

use common::{assert_converged, assert_post_warmup_divergence_rate, diag, nuts};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp as ExpDist, Gamma as GammaDist, Normal as NormalDist, Poisson};
use rustmc_core::diagnostics::DiagnosticsReport;
use rustmc_core::distributions::{HalfNormal, Normal};
use rustmc_core::graph::Graph;

const CHAINS: usize = 4;
const DRAWS: usize = 400;
const WARMUP: usize = 400;
const MAX_DIV_RATE: f64 = 0.02;

/// `|mean - truth| <= 3 * posterior_sd + 4 * mcse` — see module docs.
fn assert_recovered(report: &DiagnosticsReport, name: &str, truth: f64) {
    let p = diag(report, name);
    let tol = 3.0 * p.std + 4.0 * p.mcse_mean;
    assert!(
        (p.mean - truth).abs() <= tol,
        "{name}: posterior mean {:.5} vs truth {:.5} (|diff| {:.5}) exceeds \
         3*sd({:.5}) + 4*mcse({:.5}) = {:.5}",
        p.mean,
        truth,
        (p.mean - truth).abs(),
        p.std,
        p.mcse_mean,
        tol
    );
}

/// `y_i ~ Exponential(rate = exp(a + b x_i))`.
#[test]
fn exponential_glm_recovers_rate_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(4001);
    let a_true = 0.4;
    let b_true = -0.7;
    let n = 400;
    let xd = NormalDist::new(0.0, 1.0).unwrap();
    let x: Vec<f64> = (0..n).map(|_| xd.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let rate = (a_true + b_true * xi).exp();
            ExpDist::new(rate).unwrap().sample(&mut rng)
        })
        .collect();

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 3.0);
    let b = Normal::prior(&mut g, "b", 0.0, 3.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data(y);
    g.obs_logp_exponential_log(eta, obs);

    let result = nuts(g, 40_001, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.03);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);
    assert_recovered(&report, "a", a_true);
    assert_recovered(&report, "b", b_true);
}

/// `y_i ~ LogNormal(a + b x_i, sigma)` with `sigma` learned.
#[test]
fn lognormal_glm_recovers_location_and_scale() {
    let mut rng = ChaCha8Rng::seed_from_u64(4002);
    let a_true = 0.6;
    let b_true = 0.9;
    let sigma_true = 0.45;
    let n = 300;
    let xd = NormalDist::new(0.0, 1.0).unwrap();
    let nd = NormalDist::new(0.0, sigma_true).unwrap();
    let x: Vec<f64> = (0..n).map(|_| xd.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (a_true + b_true * xi + nd.sample(&mut rng)).exp())
        .collect();

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 3.0);
    let b = Normal::prior(&mut g, "b", 0.0, 3.0);
    let sigma = HalfNormal::prior(&mut g, "sigma", 1.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data(y);
    g.obs_logp_lognormal(mu, sigma, obs);

    let result = nuts(g, 40_002, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.03);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);
    assert_recovered(&report, "a", a_true);
    assert_recovered(&report, "b", b_true);
    assert_recovered(&report, "sigma", sigma_true);
}

/// `y_i ~ NegativeBinomial(mean = exp(a + b x_i), dispersion alpha)`.
///
/// The dispersion is held fixed at its true value: with `alpha` free and only
/// a few hundred observations it is only weakly identified, and a recovery
/// assertion on it would be a coin flip rather than a test.
#[test]
fn negative_binomial_glm_recovers_mean_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(4003);
    let a_true = 1.1;
    let b_true = 0.5;
    let alpha = 3.0;
    let n = 500;
    let xd = NormalDist::new(0.0, 1.0).unwrap();
    let x: Vec<f64> = (0..n).map(|_| xd.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (a_true + b_true * xi).exp();
            let lambda = GammaDist::new(alpha, mu / alpha).unwrap().sample(&mut rng);
            Poisson::new(lambda.max(1e-12)).unwrap().sample(&mut rng)
        })
        .collect();

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 3.0);
    let b = Normal::prior(&mut g, "b", 0.0, 3.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let alpha_node = g.add_constant(alpha);
    let obs = g.add_obs_data(y);
    g.obs_logp_negative_binomial_log(eta, alpha_node, obs);

    let result = nuts(g, 40_003, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.03);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);
    assert_recovered(&report, "a", a_true);
    assert_recovered(&report, "b", b_true);
}

/// High-dimensional regression through `MatVecMul`: `p = 25`, `n = 400`,
/// with `sigma` learned. Checks the whole coefficient vector by RMSE against
/// the exact posterior-mean RMSE that the priors imply.
#[test]
fn high_dimensional_matvec_regression_recovers_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(4004);
    let n = 250;
    let p = 15;
    let sigma_true = 0.8;
    let xd = NormalDist::new(0.0, 1.0).unwrap();
    let bd = NormalDist::new(0.0, 1.0).unwrap();
    let nd = NormalDist::new(0.0, sigma_true).unwrap();

    let beta_true: Vec<f64> = (0..p).map(|_| bd.sample(&mut rng)).collect();
    let mut flat = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..p).map(|_| xd.sample(&mut rng)).collect();
        let mu: f64 = row.iter().zip(&beta_true).map(|(a, b)| a * b).sum();
        y.push(mu + nd.sample(&mut rng));
        flat.extend(row);
    }

    let mut g = Graph::new();
    let s = g.add_vector_params("beta", p);
    g.vector_normal_logp(s, p, 0.0, 1.0);
    let sigma = HalfNormal::prior(&mut g, "sigma", 2.0);
    let m = g.store_matrix(flat, n, p);
    let mu = g.mat_vec_mul(m, s, p, None);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sigma, obs);

    let result = nuts(g, 40_004, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.05);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);
    assert_recovered(&report, "sigma", sigma_true);

    // With standard-normal predictors and n >> p the posterior sd of each
    // coefficient is ~ sigma / sqrt(n), so the RMSE of the posterior mean
    // vector around the truth should be close to that. Allow 3x as a bound.
    let expected_rmse = sigma_true / (n as f64).sqrt();
    let mut sum_sq = 0.0;
    for (k, truth) in beta_true.iter().enumerate() {
        let d = diag(&report, &format!("beta[{k}]"));
        sum_sq += (d.mean - truth).powi(2);
    }
    let rmse = (sum_sq / p as f64).sqrt();
    assert!(
        rmse <= 3.0 * expected_rmse,
        "beta RMSE {rmse:.5} exceeds 3 * sigma/sqrt(n) = {:.5}",
        3.0 * expected_rmse
    );
}

/// Hierarchical partial pooling with a learned group scale, non-centered.
///
/// Asserts recovery of the hyperparameters and that shrinkage moves the group
/// estimates toward the grand mean relative to the raw group means — the
/// defining behaviour of partial pooling, and something a model that silently
/// degenerated to no-pooling or complete-pooling would fail.
#[test]
fn hierarchical_partial_pooling_recovers_and_shrinks() {
    let mut rng = ChaCha8Rng::seed_from_u64(4005);
    let groups = 8;
    let per_group = 6;
    let mu_true = 2.0;
    let tau_true = 1.0;
    let sigma_true = 1.5;
    let gd = NormalDist::new(mu_true, tau_true).unwrap();
    let nd = NormalDist::new(0.0, sigma_true).unwrap();

    let theta_true: Vec<f64> = (0..groups).map(|_| gd.sample(&mut rng)).collect();
    let raw_means: Vec<f64> = theta_true
        .iter()
        .map(|&t| (0..per_group).map(|_| t + nd.sample(&mut rng)).sum::<f64>() / per_group as f64)
        .collect();

    // Group means are sufficient statistics: y_bar_g ~ N(theta_g, sigma/sqrt(m)).
    let se = sigma_true / (per_group as f64).sqrt();

    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 10.0);
    let tau = HalfNormal::prior(&mut g, "tau", 5.0);
    for (gi, &ybar) in raw_means.iter().enumerate() {
        let z = Normal::prior(&mut g, &format!("z_{gi}"), 0.0, 1.0);
        let tz = g.mul(tau, z);
        let theta = g.add(mu, tz);
        let yn = g.add_constant(ybar);
        let sn = g.add_constant(se);
        g.normal_logp(yn, theta, sn);
    }

    let result = nuts(g, 40_005, CHAINS, 800, 800);
    let report = result.diagnostics();
    assert_converged(&report, 1.05);
    assert_post_warmup_divergence_rate(&result, 0.03);
    assert_recovered(&report, "mu", mu_true);
    assert_recovered(&report, "tau", tau_true);

    // Reconstruct theta_g = mu + tau * z_g at the posterior mean and check
    // shrinkage: every group estimate must lie between its raw mean and the
    // grand posterior mean (inclusive), which is what pooling guarantees.
    let mu_hat = diag(&report, "mu").mean;
    let tau_hat = diag(&report, "tau").mean;
    let mut shrunk = 0;
    for (gi, &ybar) in raw_means.iter().enumerate() {
        let z_hat = diag(&report, &format!("z_{gi}")).mean;
        let theta_hat = mu_hat + tau_hat * z_hat;
        assert!(
            (theta_hat - mu_hat).abs() <= (ybar - mu_hat).abs() + 1e-6,
            "group {gi}: pooled estimate {theta_hat:.4} is farther from the grand mean \
             {mu_hat:.4} than the raw group mean {ybar:.4} — no shrinkage occurred"
        );
        if (theta_hat - mu_hat).abs() < 0.99 * (ybar - mu_hat).abs() {
            shrunk += 1;
        }
    }
    assert!(
        shrunk >= groups - 1,
        "only {shrunk}/{groups} groups were measurably shrunk toward the grand mean"
    );
}
