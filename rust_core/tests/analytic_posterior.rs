//! Analytical reference tests: models whose posterior is available in closed
//! form, compared against the sampler's output.
//!
//! These are the strongest correctness evidence in the suite. Unlike parameter
//! recovery (which only checks that the posterior is *near the truth*), these
//! check that the posterior is *the right distribution*: both its mean and its
//! spread are asserted against exact values.
//!
//! See `common/mod.rs` for the tolerance derivation.

mod common;

use common::{
    assert_converged, assert_mean_within_mcse, assert_post_warmup_divergences,
    assert_sd_within_mcse, invert, linear_gaussian_posterior, mat_vec, nuts, Rng,
};
use rustmc_core::distributions::Normal;
use rustmc_core::graph::Graph;

const CHAINS: usize = 4;
const DRAWS: usize = 600;
const WARMUP: usize = 600;

/// Conjugate Normal-Normal with known observation sd.
///
/// Target failure mode: a systematic shift or scale error in the Normal
/// log-density or in its gradient. The posterior mean and sd are exact, so a
/// bug that (say) dropped a factor of 2 in the quadratic term would move the
/// posterior sd by sqrt(2) and fail immediately.
#[test]
fn conjugate_normal_normal_matches_analytic_posterior() {
    let mut rng = Rng::new(0xC0FFEE);
    let mu_true = 1.25;
    let sigma = 0.8; // known
    let m0 = 0.0;
    let s0 = 2.0;
    let n = 40;
    let y: Vec<f64> = (0..n).map(|_| rng.normal_with(mu_true, sigma)).collect();

    // Exact posterior.
    let prec = 1.0 / (s0 * s0) + n as f64 / (sigma * sigma);
    let post_mean = (m0 / (s0 * s0) + y.iter().sum::<f64>() / (sigma * sigma)) / prec;
    let post_sd = (1.0 / prec).sqrt();

    let mut graph = Graph::new();
    let mu = Normal::prior(&mut graph, "mu", m0, s0);
    let mu_vec = graph.scalar_broadcast(mu);
    let sigma_node = graph.add_constant(sigma);
    let obs = graph.add_obs_data(y);
    graph.normal_obs_logp(mu_vec, sigma_node, obs);

    let result = nuts(graph, 20_001, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergences(&result, 0);
    assert_mean_within_mcse(&report, "mu", post_mean, 500.0, 0.0);
    assert_sd_within_mcse(&report, "mu", post_sd, 500.0, 0.0);
}

/// Conjugate Normal-Normal where the *prior* dominates (n = 3, wide sigma).
///
/// Target failure mode: the prior term being silently dropped or double
/// counted. With little data the posterior is almost the prior, so a missing
/// prior contribution shows up as a large sd error rather than a small one.
#[test]
fn conjugate_normal_normal_prior_dominated_matches_analytic_posterior() {
    let y = vec![2.0, -1.0, 0.5];
    let sigma = 5.0;
    let m0 = 1.0;
    let s0 = 0.7;
    let n = y.len();

    let prec = 1.0 / (s0 * s0) + n as f64 / (sigma * sigma);
    let post_mean = (m0 / (s0 * s0) + y.iter().sum::<f64>() / (sigma * sigma)) / prec;
    let post_sd = (1.0 / prec).sqrt();

    let mut graph = Graph::new();
    let mu = Normal::prior(&mut graph, "mu", m0, s0);
    let mu_vec = graph.scalar_broadcast(mu);
    let sigma_node = graph.add_constant(sigma);
    let obs = graph.add_obs_data(y);
    graph.normal_obs_logp(mu_vec, sigma_node, obs);

    let result = nuts(graph, 20_002, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergences(&result, 0);
    assert_mean_within_mcse(&report, "mu", post_mean, 500.0, 0.0);
    assert_sd_within_mcse(&report, "mu", post_sd, 500.0, 0.0);
}

/// Linear Gaussian regression with known noise sd, via the `FusedLinearMu` op.
///
/// Target failure mode: an indexing or scaling error in `FusedLinearMu`'s
/// forward pass or its adjoint. The full posterior mean vector and marginal
/// sds are exact.
#[test]
fn linear_gaussian_regression_fused_matches_analytic_posterior() {
    let mut rng = Rng::new(0xBEEF01);
    let n = 60;
    let sigma = 0.7;
    let beta_true = [0.9, -1.4, 0.35];
    let prior_mean = [0.0, 0.0, 0.0];
    let prior_sd = [3.0, 3.0, 3.0];

    let x1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![1.0, x1[i], x2[i]]).collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| {
            r.iter().zip(beta_true).map(|(a, b)| a * b).sum::<f64>() + rng.normal_with(0.0, sigma)
        })
        .collect();

    let exact = linear_gaussian_posterior(&rows, &y, sigma, &prior_mean, &prior_sd);

    let mut graph = Graph::new();
    let a = Normal::prior(&mut graph, "b0", prior_mean[0], prior_sd[0]);
    let b = Normal::prior(&mut graph, "b1", prior_mean[1], prior_sd[1]);
    let c = Normal::prior(&mut graph, "b2", prior_mean[2], prior_sd[2]);
    let d0 = graph.store_data_vec(vec![1.0; n]);
    let d1 = graph.store_data_vec(x1);
    let d2 = graph.store_data_vec(x2);
    let mu = graph.fused_linear_mu(vec![a, b, c], vec![d0, d1, d2], None);
    let sigma_node = graph.add_constant(sigma);
    let obs = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, sigma_node, obs);

    let result = nuts(graph, 20_003, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergences(&result, 0);
    for (k, name) in ["b0", "b1", "b2"].iter().enumerate() {
        assert_mean_within_mcse(&report, name, exact.mean[k], 400.0, 0.0);
        assert_sd_within_mcse(&report, name, exact.sd(k), 400.0, 0.0);
    }
}

/// The same exact posterior, reached through the faer-backed `MatVecMul` op
/// and the vectorized `VectorNormalLogP` prior.
///
/// Target failure mode: the matrix path disagreeing with the scalar path.
/// Because both this test and the previous one assert against the *same kind*
/// of closed form, a discrepancy localizes the bug to the matrix code.
#[test]
fn matvec_regression_matches_analytic_posterior() {
    let mut rng = Rng::new(0xBEEF02);
    let n = 60;
    let p = 4;
    let sigma = 0.6;
    let prior_sd_scalar = 2.0;
    let beta_true = [0.6, -1.1, 0.25, 0.8];

    // Row-major n x p design matrix, first column an intercept.
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = vec![1.0];
        for _ in 1..p {
            row.push(rng.normal());
        }
        rows.push(row);
    }
    let y: Vec<f64> = rows
        .iter()
        .map(|r| {
            r.iter().zip(beta_true).map(|(a, b)| a * b).sum::<f64>() + rng.normal_with(0.0, sigma)
        })
        .collect();

    let prior_mean = vec![0.0; p];
    let prior_sd = vec![prior_sd_scalar; p];
    let exact = linear_gaussian_posterior(&rows, &y, sigma, &prior_mean, &prior_sd);

    let mut graph = Graph::new();
    let beta_start = graph.add_vector_params("beta", p);
    graph.vector_normal_logp(beta_start, p, 0.0, prior_sd_scalar);
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let matrix = graph.store_matrix(flat, n, p);
    let mu = graph.mat_vec_mul(matrix, beta_start, p, None);
    let sigma_node = graph.add_constant(sigma);
    let obs = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, sigma_node, obs);

    let result = nuts(graph, 20_004, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.03);
    assert_post_warmup_divergences(&result, 0);
    for k in 0..p {
        let name = format!("beta[{k}]");
        let name = if report.params.iter().any(|q| q.name == name) {
            name
        } else {
            format!("beta_{k}")
        };
        assert_mean_within_mcse(&report, &name, exact.mean[k], 400.0, 0.0);
        assert_sd_within_mcse(&report, &name, exact.sd(k), 400.0, 0.0);
    }
}

/// Two-level linear-Gaussian hierarchy with *known* variance components, so
/// the joint posterior over (mu, theta_1..theta_J) is exactly Gaussian.
///
/// Target failure mode: the hierarchical `prior_with_nodes` path mis-wiring
/// the parent mean, or the partial-pooling shrinkage coming out at the wrong
/// strength. Shrinkage strength is entirely determined by tau vs sigma_j, so
/// an error there moves both the posterior means and their sds.
#[test]
fn hierarchical_linear_gaussian_matches_analytic_posterior() {
    let j = 6usize;
    let s0 = 5.0; // prior sd on mu
    let tau = 1.2; // known group sd
    let sigma = [1.0, 1.4, 0.8, 2.0, 1.1, 0.9]; // known within-group sds
    let y = [1.9, -0.4, 3.1, 0.2, 1.1, 2.4];

    // Joint precision over [mu, theta_0..theta_{J-1}].
    let dim = j + 1;
    let mut lambda = vec![vec![0.0; dim]; dim];
    let mut h = vec![0.0; dim];
    lambda[0][0] = 1.0 / (s0 * s0) + j as f64 / (tau * tau);
    for g in 0..j {
        lambda[0][g + 1] = -1.0 / (tau * tau);
        lambda[g + 1][0] = -1.0 / (tau * tau);
        lambda[g + 1][g + 1] = 1.0 / (tau * tau) + 1.0 / (sigma[g] * sigma[g]);
        h[g + 1] = y[g] / (sigma[g] * sigma[g]);
    }
    let cov = invert(&lambda);
    let mean = mat_vec(&cov, &h);

    let mut graph = Graph::new();
    let mu = Normal::prior(&mut graph, "mu", 0.0, s0);
    let tau_node = graph.add_constant(tau);
    for g in 0..j {
        let theta = Normal::prior_with_nodes(&mut graph, &format!("theta_{g}"), mu, tau_node);
        let yi = graph.add_constant(y[g]);
        let si = graph.add_constant(sigma[g]);
        graph.normal_logp(yi, theta, si);
    }

    let result = nuts(graph, 20_005, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.03);
    assert_post_warmup_divergences(&result, 0);
    assert_mean_within_mcse(&report, "mu", mean[0], 400.0, 0.0);
    assert_sd_within_mcse(&report, "mu", cov[0][0].sqrt(), 400.0, 0.0);
    for g in 0..j {
        let name = format!("theta_{g}");
        assert_mean_within_mcse(&report, &name, mean[g + 1], 400.0, 0.0);
        assert_sd_within_mcse(&report, &name, cov[g + 1][g + 1].sqrt(), 400.0, 0.0);
    }
}

/// Near-collinear predictors (empirical correlation ~0.999) with a proper
/// prior, so the posterior is still exactly Gaussian but extremely elongated.
///
/// Target failure mode: mass-matrix adaptation failing on a badly conditioned
/// target, producing a posterior that is too narrow along the stiff direction.
/// The tolerance widens automatically because ESS collapses; the assertion
/// still has teeth because the two marginal sds differ by orders of magnitude
/// from the well-conditioned case.
#[test]
fn near_collinear_regression_matches_analytic_posterior() {
    let mut rng = Rng::new(0xBEEF03);
    let n = 100;
    let sigma = 0.5;
    let prior_sd = [2.0, 2.0];

    let x1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let x2: Vec<f64> = x1.iter().map(|&v| v + 0.03 * rng.normal()).collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![x1[i], x2[i]]).collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| 1.0 * r[0] - 0.5 * r[1] + rng.normal_with(0.0, sigma))
        .collect();

    let exact = linear_gaussian_posterior(&rows, &y, sigma, &[0.0, 0.0], &prior_sd);

    let mut graph = Graph::new();
    let b1 = Normal::prior(&mut graph, "b1", 0.0, prior_sd[0]);
    let b2 = Normal::prior(&mut graph, "b2", 0.0, prior_sd[1]);
    let d1 = graph.store_data_vec(x1);
    let d2 = graph.store_data_vec(x2);
    let mu = graph.fused_linear_mu(vec![b1, b2], vec![d1, d2], None);
    let sigma_node = graph.add_constant(sigma);
    let obs = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, sigma_node, obs);

    let result = nuts(graph, 20_006, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.05);
    assert_post_warmup_divergences(&result, 5);
    for (k, name) in ["b1", "b2"].iter().enumerate() {
        assert_mean_within_mcse(&report, name, exact.mean[k], 200.0, 0.0);
        assert_sd_within_mcse(&report, name, exact.sd(k), 200.0, 0.0);
    }
}
