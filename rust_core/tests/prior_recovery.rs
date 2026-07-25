//! Prior-only sampling: does the sampler reproduce the distribution it was
//! told to sample from?
//!
//! A model with priors but no likelihood has a posterior that *is* the prior.
//! Its moments are known in closed form for every family the DSL supports, so
//! this is the cleanest possible end-to-end check of:
//!
//!   * the log-density algebra for each family,
//!   * the constraining transform (exp / sigmoid / bounded sigmoid), and
//!   * the Jacobian correction that goes with it.
//!
//! A missing or wrong Jacobian is the classic silent-bias bug: the sampler
//! still converges, but to a tilted version of the intended distribution.
//! `gradient_check.rs` cannot catch it (the gradient of a *wrong* density is
//! still self-consistent); this file can.
//!
//! It is also the prior-predictive validation required by `VALIDATION.md`
//! workstream 2, expressed at the parameter level where the reference values
//! are exact rather than simulated.

mod common;

use common::{
    assert_converged, assert_mean_within_mcse, assert_post_warmup_divergence_rate,
    assert_sd_within_mcse_kurtosis, diag, nuts,
};
use rustmc_core::distributions::{
    BetaDist, Exponential, Gamma, HalfNormal, LogNormal, Normal, StudentT, Uniform,
};
use rustmc_core::graph::{Graph, ParamTransform};

const CHAINS: usize = 4;
const DRAWS: usize = 1500;
const WARMUP: usize = 500;
const MIN_ESS: f64 = 800.0;
/// Log-scale parameters (HalfNormal, Gamma, Exponential, LogNormal) have a
/// doubly-exponential right tail in unconstrained space, which produces a small
/// residual divergence rate at any practical step size. Measured rate across 24
/// independent seeds for the HalfNormal case is 0.29%; 1% is a generous but
/// still meaningful ceiling. Bias, not divergence count, is what the moment
/// assertions below actually test.
const MAX_DIV_RATE: f64 = 0.01;

/// Moments of a univariate family: mean, sd, and standardized fourth moment.
struct Moments {
    mean: f64,
    sd: f64,
    kurtosis: f64,
}

fn normal_moments(mu: f64, sigma: f64) -> Moments {
    Moments {
        mean: mu,
        sd: sigma,
        kurtosis: 3.0,
    }
}

fn half_normal_moments(sigma: f64) -> Moments {
    let two_over_pi = 2.0 / std::f64::consts::PI;
    Moments {
        mean: sigma * two_over_pi.sqrt(),
        sd: sigma * (1.0 - two_over_pi).sqrt(),
        // Excess kurtosis of the half-normal is 8(pi - 3)/(pi - 2)^2.
        kurtosis: 3.0 + 8.0 * (std::f64::consts::PI - 3.0) / (std::f64::consts::PI - 2.0).powi(2),
    }
}

fn log_normal_moments(mu: f64, sigma: f64) -> Moments {
    let s2 = sigma * sigma;
    let mean = (mu + 0.5 * s2).exp();
    let var = ((s2).exp() - 1.0) * (2.0 * mu + s2).exp();
    Moments {
        mean,
        sd: var.sqrt(),
        kurtosis: (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 3.0,
    }
}

fn exponential_moments(rate: f64) -> Moments {
    Moments {
        mean: 1.0 / rate,
        sd: 1.0 / rate,
        kurtosis: 9.0,
    }
}

fn gamma_moments(alpha: f64, beta: f64) -> Moments {
    Moments {
        mean: alpha / beta,
        sd: alpha.sqrt() / beta,
        kurtosis: 3.0 + 6.0 / alpha,
    }
}

fn beta_moments(a: f64, b: f64) -> Moments {
    let s = a + b;
    let var = a * b / (s * s * (s + 1.0));
    let excess =
        6.0 * ((a - b).powi(2) * (s + 1.0) - a * b * (s + 2.0)) / (a * b * (s + 2.0) * (s + 3.0));
    Moments {
        mean: a / s,
        sd: var.sqrt(),
        kurtosis: 3.0 + excess,
    }
}

fn uniform_moments(lower: f64, upper: f64) -> Moments {
    Moments {
        mean: 0.5 * (lower + upper),
        sd: (upper - lower) / 12.0_f64.sqrt(),
        kurtosis: 1.8,
    }
}

/// StudentT scaled by `sigma`; requires `nu > 4` for a finite fourth moment.
fn student_t_moments(nu: f64, mu: f64, sigma: f64) -> Moments {
    Moments {
        mean: mu,
        sd: sigma * (nu / (nu - 2.0)).sqrt(),
        kurtosis: 3.0 + 6.0 / (nu - 4.0),
    }
}

fn check_prior(label: &str, graph: Graph, seed: u64, name: &str, m: &Moments) {
    let result = nuts(graph, seed, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);
    assert_mean_within_mcse(&report, name, m.mean, MIN_ESS, 0.0);
    assert_sd_within_mcse_kurtosis(&report, name, m.sd, m.kurtosis, MIN_ESS, 0.0);
    let p = diag(&report, name);
    println!(
        "{label:<28} mean {:>9.5} (exact {:>9.5})  sd {:>9.5} (exact {:>9.5})  ess {:>7.0}",
        p.mean, m.mean, p.std, m.sd, p.ess_bulk
    );
}

#[test]
fn normal_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    Normal::prior(&mut g, "x", 0.3, 1.4);
    check_prior(
        "Normal(0.3, 1.4)",
        g,
        30_001,
        "x",
        &normal_moments(0.3, 1.4),
    );
}

/// Exp transform: a missing `+raw` Jacobian would bias this toward zero.
#[test]
fn half_normal_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    HalfNormal::prior(&mut g, "x", 1.1);
    check_prior("HalfNormal(1.1)", g, 30_002, "x", &half_normal_moments(1.1));
}

#[test]
fn log_normal_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    LogNormal::prior(&mut g, "x", -0.2, 0.5);
    check_prior(
        "LogNormal(-0.2, 0.5)",
        g,
        30_003,
        "x",
        &log_normal_moments(-0.2, 0.5),
    );
}

#[test]
fn exponential_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    Exponential::prior(&mut g, "x", 2.5);
    check_prior(
        "Exponential(rate=2.5)",
        g,
        30_004,
        "x",
        &exponential_moments(2.5),
    );
}

#[test]
fn gamma_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    Gamma::prior(&mut g, "x", 2.3, 1.7);
    check_prior(
        "Gamma(2.3, rate=1.7)",
        g,
        30_005,
        "x",
        &gamma_moments(2.3, 1.7),
    );
}

/// Sigmoid transform: the Jacobian is `log x + log(1 - x)`.
#[test]
fn beta_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    BetaDist::prior(&mut g, "x", 2.0, 3.5);
    check_prior("Beta(2.0, 3.5)", g, 30_006, "x", &beta_moments(2.0, 3.5));
}

/// Bounded-sigmoid transform. The Uniform log-density is constant inside the
/// support, so the *entire* shape of this posterior comes from the Jacobian:
/// if the Jacobian were dropped the draws would pile up at the boundaries.
#[test]
fn uniform_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    Uniform::prior(&mut g, "x", -2.0, 3.0);
    check_prior(
        "Uniform(-2, 3)",
        g,
        30_007,
        "x",
        &uniform_moments(-2.0, 3.0),
    );
}

#[test]
fn student_t_prior_reproduces_its_own_distribution() {
    let mut g = Graph::new();
    StudentT::prior(&mut g, "x", 8.0, 0.5, 1.3);
    check_prior(
        "StudentT(nu=8, 0.5, 1.3)",
        g,
        30_008,
        "x",
        &student_t_moments(8.0, 0.5, 1.3),
    );
}

// ── Vectorized prior ops ───────────────────────────────────────────────────
//
// The `Vector*LogP` ops are a separate implementation from the scalar
// distributions (they operate directly on the parameter slice and accumulate
// gradients without going through the graph). Their marginals must match the
// same closed forms.

fn check_vector_prior(label: &str, graph: Graph, seed: u64, n: usize, m: &Moments) {
    let result = nuts(graph, seed, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);
    for k in 0..n {
        let bracket = format!("v[{k}]");
        let name = if report.params.iter().any(|q| q.name == bracket) {
            bracket
        } else {
            format!("v_{k}")
        };
        assert_mean_within_mcse(&report, &name, m.mean, MIN_ESS, 0.0);
        assert_sd_within_mcse_kurtosis(&report, &name, m.sd, m.kurtosis, MIN_ESS, 0.0);
    }
    println!("{label:<28} all {n} marginals matched closed-form moments");
}

#[test]
fn vector_normal_prior_reproduces_its_own_distribution() {
    let n = 3;
    let mut g = Graph::new();
    let s = g.add_vector_params("v", n);
    g.vector_normal_logp(s, n, 0.4, 1.6);
    check_vector_prior(
        "VectorNormal(0.4, 1.6)",
        g,
        31_001,
        n,
        &normal_moments(0.4, 1.6),
    );
}

#[test]
fn vector_half_normal_prior_reproduces_its_own_distribution() {
    let n = 3;
    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform("v", n, ParamTransform::Exp);
    g.vector_half_normal_logp(s, n, 1.3);
    check_vector_prior(
        "VectorHalfNormal(1.3)",
        g,
        31_002,
        n,
        &half_normal_moments(1.3),
    );
}

#[test]
fn vector_gamma_prior_reproduces_its_own_distribution() {
    let n = 3;
    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform("v", n, ParamTransform::Exp);
    g.vector_gamma_logp(s, n, 2.5, 1.4);
    check_vector_prior(
        "VectorGamma(2.5, 1.4)",
        g,
        31_003,
        n,
        &gamma_moments(2.5, 1.4),
    );
}

#[test]
fn vector_beta_prior_reproduces_its_own_distribution() {
    let n = 3;
    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform("v", n, ParamTransform::Sigmoid);
    g.vector_beta_logp(s, n, 2.2, 3.1);
    check_vector_prior(
        "VectorBeta(2.2, 3.1)",
        g,
        31_004,
        n,
        &beta_moments(2.2, 3.1),
    );
}

#[test]
fn vector_uniform_prior_reproduces_its_own_distribution() {
    let n = 3;
    let mut g = Graph::new();
    let s = g.add_vector_params_with_transform(
        "v",
        n,
        ParamTransform::BoundedSigmoid {
            lower: -1.0,
            upper: 4.0,
        },
    );
    g.vector_uniform_logp(s, n, -1.0, 4.0);
    check_vector_prior(
        "VectorUniform(-1, 4)",
        g,
        31_005,
        n,
        &uniform_moments(-1.0, 4.0),
    );
}

#[test]
fn vector_student_t_prior_reproduces_its_own_distribution() {
    let n = 3;
    let mut g = Graph::new();
    let s = g.add_vector_params("v", n);
    g.vector_student_t_logp(s, n, 8.0, -0.3, 1.1);
    check_vector_prior(
        "VectorStudentT(8, -0.3, 1.1)",
        g,
        31_006,
        n,
        &student_t_moments(8.0, -0.3, 1.1),
    );
}

/// A hierarchical prior with no data: `theta ~ N(mu, tau)` with `mu` and `tau`
/// themselves given priors. Marginalizing analytically, `theta` has mean 0 and
/// variance `Var(mu) + E[tau^2]`, and its distribution is a scale mixture, so
/// only the mean and variance are asserted (with the kurtosis of the mixture
/// computed exactly from the component moments).
#[test]
fn hierarchical_prior_marginal_matches_closed_form() {
    let mu_sd = 1.5;
    let tau_sd = 0.9; // tau ~ HalfNormal(tau_sd)

    let mut g = Graph::new();
    let mu = Normal::prior(&mut g, "mu", 0.0, mu_sd);
    let tau = HalfNormal::prior(&mut g, "tau", tau_sd);
    // Non-centered so the geometry is benign; the marginal is unchanged.
    let z = Normal::prior(&mut g, "z", 0.0, 1.0);
    let tz = g.mul(tau, z);
    let _theta = g.add(mu, tz);

    let result = nuts(g, 31_007, CHAINS, DRAWS, WARMUP);
    let report = result.diagnostics();
    assert_converged(&report, 1.02);
    assert_post_warmup_divergence_rate(&result, MAX_DIV_RATE);

    // Component marginals are exact.
    assert_mean_within_mcse(&report, "mu", 0.0, MIN_ESS, 0.0);
    assert_sd_within_mcse_kurtosis(&report, "mu", mu_sd, 3.0, MIN_ESS, 0.0);
    assert_mean_within_mcse(&report, "z", 0.0, MIN_ESS, 0.0);
    assert_sd_within_mcse_kurtosis(&report, "z", 1.0, 3.0, MIN_ESS, 0.0);
    let tau_m = half_normal_moments(tau_sd);
    assert_mean_within_mcse(&report, "tau", tau_m.mean, MIN_ESS, 0.0);
    assert_sd_within_mcse_kurtosis(&report, "tau", tau_m.sd, tau_m.kurtosis, MIN_ESS, 0.0);
}
