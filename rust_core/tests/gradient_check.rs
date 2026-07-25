//! Finite-difference validation of the reverse-mode gradient.
//!
//! Every statistical claim in this repository rests on `grad_logp` returning
//! the exact gradient of `eval_logp`. If it does not, NUTS still runs, still
//! converges, and still produces a plausible-looking posterior — it is simply
//! the posterior of a *different* model. That failure mode is invisible to
//! parameter-recovery tests when the error is small, so it is checked directly
//! here.
//!
//! These tests are deterministic and fast (no sampling), so they run on every
//! `cargo test`.
//!
//! # Tolerance
//!
//! Central differences have truncation error O(h^2 * |f'''|) and roundoff
//! O(eps * |f| / h). With `h = 1e-5` and f64 (eps ~ 2.2e-16) the roundoff floor
//! is ~1e-11 * |f| and truncation ~1e-10 * |f'''|. We therefore require
//! agreement to a relative tolerance of 1e-5 with an absolute floor of 1e-6,
//! which is loose enough to never flake and ~5 orders of magnitude tighter
//! than any plausible algebra error (a dropped factor of 2, a sign flip, a
//! missing Jacobian term).

mod common;

use rustmc_core::autodiff::{eval_logp, grad_logp};
use rustmc_core::distributions::{
    BetaDist, Exponential, Gamma, HalfNormal, LogNormal, Normal, StudentT, Uniform,
};
use rustmc_core::graph::Graph;

const H: f64 = 1e-5;
const REL_TOL: f64 = 1e-5;
const ABS_TOL: f64 = 1e-6;

/// Compare `grad_logp` against central finite differences of `eval_logp`.
fn check_gradient(label: &str, graph: &Graph, point: &[f64]) {
    let (logp, analytic) = grad_logp(graph, point);
    assert!(
        logp.is_finite(),
        "{label}: logp is not finite at {point:?} (got {logp})"
    );
    assert_eq!(
        analytic.len(),
        point.len(),
        "{label}: gradient length {} != parameter count {}",
        analytic.len(),
        point.len()
    );

    for i in 0..point.len() {
        let mut up = point.to_vec();
        let mut down = point.to_vec();
        let h = H * point[i].abs().max(1.0);
        up[i] += h;
        down[i] -= h;
        let numeric = (eval_logp(graph, &up) - eval_logp(graph, &down)) / (2.0 * h);
        // Cancellation floor of the difference quotient: the two logp values
        // are each known to ~eps*|logp|, so the quotient carries an
        // irreducible error of ~eps*|logp|/h. At extreme parameter values
        // |logp| can reach 1e8, which dominates every other error term.
        let fd_noise = 8.0 * f64::EPSILON * logp.abs() / h;
        let tol = REL_TOL * numeric.abs().max(analytic[i].abs()) + ABS_TOL + fd_noise;
        assert!(
            (analytic[i] - numeric).abs() <= tol,
            "{label}: d logp / d p[{i}] analytic {:.12e} vs finite-difference {:.12e} \
             (|diff| {:.3e} > tol {:.3e}); full gradient {:?}",
            analytic[i],
            numeric,
            (analytic[i] - numeric).abs(),
            tol,
            analytic
        );
    }
}

fn design(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = common::Rng::new(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    (x, y)
}

// ── Scalar prior families: transform + Jacobian correctness ────────────────

#[test]
fn scalar_prior_gradients_match_finite_differences() {
    // Each entry: (label, builder, evaluation points in *unconstrained* space)
    let points = [-1.7_f64, -0.4, 0.0, 0.6, 1.9];

    let mut g = Graph::new();
    Normal::prior(&mut g, "x", 0.3, 1.4);
    for &p in &points {
        check_gradient("Normal prior", &g, &[p]);
    }

    let mut g = Graph::new();
    HalfNormal::prior(&mut g, "x", 1.1);
    for &p in &points {
        check_gradient("HalfNormal prior (exp transform)", &g, &[p]);
    }

    let mut g = Graph::new();
    LogNormal::prior(&mut g, "x", -0.2, 0.9);
    for &p in &points {
        check_gradient("LogNormal prior", &g, &[p]);
    }

    let mut g = Graph::new();
    Exponential::prior(&mut g, "x", 2.5);
    for &p in &points {
        check_gradient("Exponential prior", &g, &[p]);
    }

    let mut g = Graph::new();
    Gamma::prior(&mut g, "x", 2.3, 1.7);
    for &p in &points {
        check_gradient("Gamma prior", &g, &[p]);
    }

    let mut g = Graph::new();
    BetaDist::prior(&mut g, "x", 2.0, 3.5);
    for &p in &points {
        check_gradient("Beta prior (sigmoid transform)", &g, &[p]);
    }

    let mut g = Graph::new();
    Uniform::prior(&mut g, "x", -2.0, 3.0);
    for &p in &points {
        check_gradient("Uniform prior (bounded sigmoid transform)", &g, &[p]);
    }

    let mut g = Graph::new();
    StudentT::prior(&mut g, "x", 4.0, 0.5, 1.3);
    for &p in &points {
        check_gradient("StudentT prior", &g, &[p]);
    }
}

// ── Vectorized prior families ──────────────────────────────────────────────

#[test]
fn vector_prior_gradients_match_finite_differences() {
    let point = [-1.2_f64, 0.3, 0.9, -0.05];

    let mut g = Graph::new();
    let s = g.add_vector_params("v", 4);
    g.vector_normal_logp(s, 4, 0.4, 1.6);
    check_gradient("VectorNormalLogP", &g, &point);

    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform("v", 4, rustmc_core::graph::ParamTransform::Exp);
    g.vector_half_normal_logp(s, 4, 1.3);
    check_gradient("VectorHalfNormalLogP", &g, &point);

    let mut g = Graph::new();
    let s = g.add_vector_params("v", 4);
    g.vector_student_t_logp(s, 4, 5.0, -0.3, 1.1);
    check_gradient("VectorStudentTLogP", &g, &point);

    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform("v", 4, rustmc_core::graph::ParamTransform::Exp);
    g.vector_gamma_logp(s, 4, 2.5, 1.4);
    check_gradient("VectorGammaLogP", &g, &point);

    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform("v", 4, rustmc_core::graph::ParamTransform::Sigmoid);
    g.vector_beta_logp(s, 4, 2.2, 3.1);
    check_gradient("VectorBetaLogP", &g, &point);

    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform(
        "v",
        4,
        rustmc_core::graph::ParamTransform::BoundedSigmoid {
            lower: -1.0,
            upper: 4.0,
        },
    );
    g.vector_uniform_logp(s, 4, -1.0, 4.0);
    check_gradient("VectorUniformLogP", &g, &point);
}

// ── Observation families ───────────────────────────────────────────────────

#[test]
fn observation_family_gradients_match_finite_differences() {
    let n = 25;
    let (x, _) = design(n, 7);
    let point = [0.4_f64, -0.7];

    // Normal
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x.clone());
    let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let s = g.add_constant(0.9);
    let (_, yy) = design(n, 8);
    let obs = g.add_obs_data(yy);
    g.normal_obs_logp(mu, s, obs);
    check_gradient("Normal observation", &g, &point);

    // BernoulliLogit
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x.clone());
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data((0..n).map(|i| (i % 2) as f64).collect());
    g.obs_logp_bernoulli_logit(eta, obs);
    check_gradient("BernoulliLogit observation", &g, &point);

    // PoissonLog
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x.clone());
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data((0..n).map(|i| (i % 5) as f64).collect());
    g.obs_logp_poisson_log(eta, obs);
    check_gradient("PoissonLog observation", &g, &point);

    // ExponentialLog
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x.clone());
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data((0..n).map(|i| 0.3 + 0.1 * i as f64).collect());
    g.obs_logp_exponential_log(eta, obs);
    check_gradient("ExponentialLog observation", &g, &point);

    // LogNormal
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x.clone());
    let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let s = g.add_constant(0.7);
    let obs = g.add_obs_data((0..n).map(|i| 0.5 + 0.2 * i as f64).collect());
    g.obs_logp_lognormal(mu, s, obs);
    check_gradient("LogNormal observation", &g, &point);

    // NegativeBinomialLog
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x.clone());
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let alpha = g.add_constant(2.5);
    let obs = g.add_obs_data((0..n).map(|i| (i % 7) as f64).collect());
    g.obs_logp_negative_binomial_log(eta, alpha, obs);
    check_gradient("NegativeBinomialLog observation", &g, &point);
}

/// Observation families with a *learned* auxiliary parameter (sigma / alpha),
/// which exercises the adjoint path through `aux` rather than a constant node.
#[test]
fn learned_scale_gradients_match_finite_differences() {
    let n = 20;
    let (x, y) = design(n, 11);

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let sigma = HalfNormal::prior(&mut g, "sigma", 1.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let mu = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sigma, obs);
    for point in [[0.3, -0.5, 0.2], [-1.1, 0.8, -0.9], [0.0, 0.0, 1.4]] {
        check_gradient("Normal observation with learned sigma", &g, &point);
    }
}

// ── Structural ops ─────────────────────────────────────────────────────────

#[test]
fn matvec_gradients_match_finite_differences() {
    let n = 30;
    let p = 5;
    let mut rng = common::Rng::new(13);
    let flat: Vec<f64> = (0..n * p).map(|_| rng.normal()).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

    // Without intercept.
    let mut g = Graph::new();
    let s = g.add_vector_params("beta", p);
    g.vector_normal_logp(s, p, 0.0, 1.5);
    let m = g.store_matrix(flat.clone(), n, p);
    let mu = g.mat_vec_mul(m, s, p, None);
    let sig = g.add_constant(0.8);
    let obs = g.add_obs_data(y.clone());
    g.normal_obs_logp(mu, sig, obs);
    check_gradient("MatVecMul", &g, &[0.4, -0.9, 0.15, 1.1, -0.3]);

    // With a learned intercept declared *after* the vector block.
    let mut g = Graph::new();
    let s = g.add_vector_params("beta", p);
    g.vector_normal_logp(s, p, 0.0, 1.5);
    let icpt = Normal::prior(&mut g, "icpt", 0.0, 3.0);
    let m = g.store_matrix(flat, n, p);
    let mu = g.mat_vec_mul(m, s, p, Some(icpt));
    let sig = g.add_constant(0.8);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sig, obs);
    check_gradient(
        "MatVecMul with intercept",
        &g,
        &[0.4, -0.9, 0.15, 1.1, -0.3, 0.6],
    );
}

#[test]
fn elementwise_op_gradients_match_finite_differences() {
    let n = 12;
    let (x, y) = design(n, 17);

    // ScalarMulData + ScalarBroadcastAdd + VectorAdd, non-fused path.
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let b = Normal::prior(&mut g, "b", 0.0, 2.0);
    let xd = g.add_data("x", x.clone());
    let bx = g.scalar_mul_data(b, xd);
    let x2d = g.add_data("x2", x.iter().map(|v| v * v).collect());
    let ax2 = g.scalar_mul_data(a, x2d);
    let both = g.vector_add(bx, ax2);
    let mu = g.scalar_broadcast_add(a, both);
    let sig = g.add_constant(1.0);
    let obs = g.add_obs_data(y.clone());
    g.normal_obs_logp(mu, sig, obs);
    check_gradient(
        "ScalarMulData / VectorAdd / ScalarBroadcastAdd",
        &g,
        &[0.7, -0.4],
    );

    // ScalarBroadcast on its own.
    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 2.0);
    let mu = g.scalar_broadcast(a);
    let sig = g.add_constant(1.2);
    let obs = g.add_obs_data(y);
    g.normal_obs_logp(mu, sig, obs);
    check_gradient("ScalarBroadcast", &g, &[0.55]);

    // Arithmetic / transcendental chain: exp, log, sigmoid, square, div, sub, neg.
    let mut g = Graph::new();
    let u = Normal::prior(&mut g, "u", 0.0, 2.0);
    let v = Normal::prior(&mut g, "v", 0.0, 2.0);
    let two = g.add_constant(2.0);
    let e = g.exp(u);
    let sq = g.square(v);
    let sum = g.add(e, sq);
    let lg = g.log(sum);
    let sg = g.sigmoid(v);
    let dv = g.div(lg, two);
    let sb = g.sub(dv, sg);
    let ng = g.neg(sb);
    let ml = g.mul(ng, e);
    let one = g.add_constant(1.0);
    let zero = g.add_constant(0.0);
    g.normal_logp(ml, zero, one);
    for point in [[0.2, 0.9], [-0.8, -1.3], [1.4, 0.05]] {
        check_gradient("arithmetic chain", &g, &point);
    }
}

#[test]
fn hierarchical_gradients_match_finite_differences() {
    // Centered: theta_j ~ N(mu, tau), tau itself a HalfNormal parameter.
    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 5.0);
    let tau = HalfNormal::prior(&mut g, "tau", 2.0);
    let y = [1.2_f64, -0.3, 2.4];
    let sd = [1.0_f64, 1.5, 0.8];
    for (j, (&yj, &sj)) in y.iter().zip(sd.iter()).enumerate() {
        let theta = Normal::prior_with_nodes(&mut g, &format!("theta_{j}"), mu, tau);
        let yn = g.add_constant(yj);
        let sn = g.add_constant(sj);
        g.normal_logp(yn, theta, sn);
    }
    for point in [
        [0.5, 0.1, 0.9, -0.2, 1.3],
        [-1.4, -0.9, 0.0, 0.7, -0.5],
        [2.0, 1.2, -1.1, 0.4, 0.8],
    ] {
        check_gradient("centered hierarchy with learned tau", &g, &point);
    }

    // Non-centered: theta = mu + tau * z.
    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 5.0);
    let tau = HalfNormal::prior(&mut g, "tau", 2.0);
    for (j, (&yj, &sj)) in y.iter().zip(sd.iter()).enumerate() {
        let z = Normal::prior(&mut g, &format!("z_{j}"), 0.0, 1.0);
        let tz = g.mul(tau, z);
        let theta = g.add(mu, tz);
        let yn = g.add_constant(yj);
        let sn = g.add_constant(sj);
        g.normal_logp(yn, theta, sn);
    }
    for point in [[0.5, 0.1, 0.9, -0.2, 1.3], [-1.4, -0.9, 0.0, 0.7, -0.5]] {
        check_gradient("non-centered hierarchy", &g, &point);
    }
}

/// The gradient must stay exact at extreme-but-valid parameter values, where
/// naive implementations of `log(1 + exp(x))` or `log(sigmoid(x))` overflow.
#[test]
fn gradients_stay_exact_at_extreme_parameter_values() {
    let n = 15;
    let (x, _) = design(n, 23);

    let mut g = Graph::new();
    let a = Normal::prior(&mut g, "a", 0.0, 10.0);
    let b = Normal::prior(&mut g, "b", 0.0, 10.0);
    let d0 = g.store_data_vec(vec![1.0; n]);
    let d1 = g.store_data_vec(x);
    let eta = g.fused_linear_mu(vec![a, b], vec![d0, d1], None);
    let obs = g.add_obs_data((0..n).map(|i| (i % 2) as f64).collect());
    g.obs_logp_bernoulli_logit(eta, obs);
    // Linear predictors reaching +-30 push sigmoid to within 1e-13 of 0 or 1.
    for point in [[25.0, 0.0], [-25.0, 0.0], [0.0, 20.0], [0.0, -20.0]] {
        check_gradient("BernoulliLogit at extreme eta", &g, &point);
    }

    // Very small and very large learned sigma.
    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, 10.0);
    let sigma = HalfNormal::prior(&mut g, "sigma", 5.0);
    let mu_vec = g.scalar_broadcast(mu);
    let obs = g.add_obs_data(vec![0.1, -0.2, 0.05, 0.0, 0.3]);
    g.normal_obs_logp(mu_vec, sigma, obs);
    // raw = log(sigma): -9 => sigma ~ 1.2e-4; +9 => sigma ~ 8100.
    for point in [[0.05, -9.0], [0.05, 9.0], [0.0, 0.0]] {
        check_gradient("Normal observation at extreme sigma", &g, &point);
    }
}
