//! Targets with exactly known moments, and the exactness gate applied to fits
//! of them. Shared by `exactness.rs` (the gate itself, its false-alarm study
//! and its cost report) and `exactness_power.rs` (the gate against
//! deliberately broken samplers).
//!
//! The gate, per coordinate of a fit at the shipped defaults:
//!
//! - no divergent transitions;
//! - bulk ESS at least [`MIN_ESS_BULK`], so the MCSEs below are estimated from
//!   enough effective draws to trust;
//! - mean: `|mean - exact mean| < MAX_MCSE * mcse_mean`;
//! - variance: the draws of `(x - exact mean)^2` have expectation exactly the
//!   exact variance, so their mean is held to the same bound, with an MCSE
//!   estimated from those squared draws' own effective sample size. That is
//!   the ESS-derived variance tolerance: `MAX_MCSE * sd((x - m)^2) /
//!   sqrt(ess((x - m)^2))`.
//!
//! and once per fit, on the whole draw:
//!
//! - squared norm: every target here is `w = A x - c ~ Normal(0, I)`, so
//!   `|w|^2` (minus twice the log density, up to a constant) has expectation
//!   exactly the dimension `d`. Its draws' mean is held to the same 4-MCSE
//!   bound. The per-coordinate checks see a bias only where it is large in
//!   the original coordinates; this one adds up a bias spread over every
//!   whitened direction, including the stiff ones a too-large step size
//!   distorts first, against noise that grows only as `sqrt(2d)`.
// Each test binary uses a different part of this module.
#![allow(dead_code)]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rustmc_core::diagnostics::compute_diagnostics;
use rustmc_core::graph::Graph;
use rustmc_core::sampler::{sample, SampleResult, SamplerConfig, SamplerType};
use rustmc_core::target::{sample_target, LogDensity};

/// Bound on every standardized error, in MCSEs. A correct sampler with a
/// correct MCSE crosses it on a given check with probability
/// [`NOMINAL_EXCEEDANCE`].
pub const MAX_MCSE: f64 = 4.0;

/// `P(|Z| >= 4)` for a standard normal `Z`: `erfc(4 / sqrt(2))`.
pub const NOMINAL_EXCEEDANCE: f64 = 6.334_248_366_623_996e-5;

/// Minimum bulk ESS for every coordinate, as in the release gate.
pub const MIN_ESS_BULK: f64 = 400.0;

/// Which sampling entry point a fit goes through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// `target::sample_target` with a hand-written [`LogDensity`]: every
    /// parameter is a scalar, so `MetricKind::Auto` adapts a diagonal metric.
    Custom,
    /// `sampler::sample` on a [`Graph`] holding one vector parameter: the
    /// autodiff evaluator, and `MetricKind::Auto` may choose a dense block.
    Graph,
}

impl Path {
    pub fn name(self) -> &'static str {
        match self {
            Path::Custom => "custom target",
            Path::Graph => "graph",
        }
    }
}

/// A Gaussian target given as `w = A x - c ~ Normal(0, I)` for an invertible
/// `A` (row-major, `dimension` square), with its exact marginal moments.
/// Both paths evaluate the same density: the custom target directly, the
/// graph as a `mat_vec_mul` node observed at `c` with unit noise.
pub struct GaussianTarget {
    pub name: &'static str,
    pub a: Vec<f64>,
    pub c: Vec<f64>,
    pub means: Vec<f64>,
    pub sds: Vec<f64>,
    /// When `Some`, the graph path builds this graph instead of the
    /// `A x - c` one (a regression written the way a user writes one).
    graph: Option<Graph>,
}

impl GaussianTarget {
    pub fn dimension(&self) -> usize {
        self.means.len()
    }

    /// Independent normals with distinct means and scales (0.5 to 2.0, a
    /// four-fold spread the metric must adapt to), so a sampler that mixes up
    /// coordinates, or adapts one shared scale, is caught.
    pub fn independent_normals(dimension: usize) -> Self {
        let means: Vec<f64> = (0..dimension)
            .map(|i| (i as f64 * 0.37).sin() * 3.0)
            .collect();
        let sds: Vec<f64> = (0..dimension)
            .map(|i| 0.5 + 1.5 * i as f64 / (dimension - 1) as f64)
            .collect();
        let mut a = vec![0.0; dimension * dimension];
        let mut c = vec![0.0; dimension];
        for i in 0..dimension {
            a[i * dimension + i] = 1.0 / sds[i];
            c[i] = means[i] / sds[i];
        }
        Self {
            name: "independent normals",
            a,
            c,
            means,
            sds,
            graph: None,
        }
    }

    /// A Gaussian whose correlation matrix is `rho^|i - j|` (a stationary
    /// AR(1) correlation) with per-coordinate scales 1, 1.5 and 2 repeating.
    /// The whitening map is bidiagonal: `w_0 = z_0` and
    /// `w_i = (z_i - rho z_(i-1)) / sqrt(1 - rho^2)` with
    /// `z = (x - mean) / sd`. A diagonal metric cannot remove this
    /// correlation. At 50 dimensions and `rho = 0.95` the correlation
    /// matrix's condition number is 989 and the covariance's 1481.
    pub fn correlated(dimension: usize, rho: f64) -> Self {
        let means: Vec<f64> = (0..dimension).map(|i| 1.0 - 0.05 * i as f64).collect();
        let sds: Vec<f64> = (0..dimension).map(|i| 1.0 + 0.5 * (i % 3) as f64).collect();
        let innovation = (1.0 - rho * rho).sqrt();
        let mut a = vec![0.0; dimension * dimension];
        for i in 0..dimension {
            if i == 0 {
                a[0] = 1.0 / sds[0];
            } else {
                a[i * dimension + i] = 1.0 / (sds[i] * innovation);
                a[i * dimension + i - 1] = -rho / (sds[i - 1] * innovation);
            }
        }
        let c = mat_vec(&a, &means, dimension);
        Self {
            name: "correlated Gaussian",
            a,
            c,
            means,
            sds,
            graph: None,
        }
    }

    /// Bayesian linear regression with known noise: `beta ~ Normal(0, tau^2 I)`
    /// and `y ~ Normal(X beta, sigma^2 I)`. The posterior is Gaussian with
    /// precision `P = X'X / sigma^2 + I / tau^2` and mean
    /// `P^-1 X'y / sigma^2`, computed here in closed form. The design has
    /// columns on scales from 0.2 to 3 sharing a common factor, so the
    /// posterior is correlated and its scales differ by more than an order of
    /// magnitude. The graph path builds it as a user would: a vector
    /// parameter with a `vector_normal_logp` prior, `mat_vec_mul` and a
    /// Normal likelihood with a constant scale.
    pub fn conjugate_regression() -> Self {
        let (rows, p, tau, sigma) = (100, 10, 2.0, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(20_261_014);
        let mut normal = || -> f64 { StandardNormal.sample(&mut rng) };
        let scales: Vec<f64> = (0..p)
            .map(|j| 0.2 * (15.0_f64).powf(j as f64 / (p - 1) as f64))
            .collect();
        let mut x = vec![0.0; rows * p];
        for i in 0..rows {
            let common = normal();
            for j in 0..p {
                x[i * p + j] = scales[j] * (normal() + 0.6 * common);
            }
        }
        let beta_true: Vec<f64> = (0..p).map(|j| 1.5 * ((j as f64) * 0.9).cos()).collect();
        let y: Vec<f64> = mat_vec(&x, &beta_true, p)
            .into_iter()
            .map(|mu| mu + sigma * normal())
            .collect();

        // Posterior precision and the right-hand side X'y / sigma^2.
        let mut precision = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for i in 0..rows {
            for j in 0..p {
                rhs[j] += x[i * p + j] * y[i] / (sigma * sigma);
                for k in 0..p {
                    precision[j * p + k] += x[i * p + j] * x[i * p + k] / (sigma * sigma);
                }
            }
        }
        for j in 0..p {
            precision[j * p + j] += 1.0 / (tau * tau);
        }
        let covariance = spd_inverse(&precision, p);
        let means = mat_vec(&covariance, &rhs, p);
        let sds = (0..p).map(|j| covariance[j * p + j].sqrt()).collect();

        // The same density as `A beta - c ~ Normal(0, I)` for the custom
        // target: A = U, the upper Cholesky factor of P, and c = U^-T X'y /
        // sigma^2, so that |U beta - c|^2 = beta'P beta - 2 beta'X'y / sigma^2
        // + const.
        let lower = cholesky(&precision, p);
        let mut a = vec![0.0; p * p];
        for j in 0..p {
            for k in 0..p {
                a[j * p + k] = lower[k * p + j];
            }
        }
        let c = forward_substitute(&lower, &rhs, p);

        let mut graph = Graph::new();
        let beta = graph.add_vector_params("beta", p);
        graph.vector_normal_logp(beta, p, 0.0, tau);
        let design = graph.store_matrix(x, rows, p);
        let mu = graph.mat_vec_mul(design, beta, p, None);
        let scale = graph.add_constant(sigma);
        let observed = graph.add_obs_data(y);
        graph.normal_obs_logp(mu, scale, observed);

        Self {
            name: "conjugate regression",
            a,
            c,
            means,
            sds,
            graph: Some(graph),
        }
    }

    /// The graph the graph path samples.
    pub fn graph(&self) -> Graph {
        if let Some(graph) = &self.graph {
            return graph.clone();
        }
        let n = self.dimension();
        let mut graph = Graph::new();
        let x = graph.add_vector_params("x", n);
        let matrix = graph.store_matrix(self.a.clone(), n, n);
        let w = graph.mat_vec_mul(matrix, x, n, None);
        let unit = graph.add_constant(1.0);
        let observed = graph.add_obs_data(self.c.clone());
        graph.normal_obs_logp(w, unit, observed);
        graph
    }
}

impl LogDensity for GaussianTarget {
    fn dimension(&self) -> usize {
        self.means.len()
    }

    fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
        let n = q.len();
        let mut residual = mat_vec(&self.a, q, n);
        for (r, c) in residual.iter_mut().zip(&self.c) {
            *r -= c;
        }
        gradient.fill(0.0);
        for i in 0..n {
            for j in 0..n {
                gradient[j] -= self.a[i * n + j] * residual[i];
            }
        }
        Ok(-0.5 * residual.iter().map(|r| r * r).sum::<f64>())
    }
}

fn mat_vec(a: &[f64], x: &[f64], cols: usize) -> Vec<f64> {
    a.chunks(cols)
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

/// Lower Cholesky factor of a symmetric positive definite `n x n` matrix.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(sum > 0.0, "matrix is not positive definite");
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    l
}

/// Solve `L v = b` for lower-triangular `L`.
fn forward_substitute(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * v[k];
        }
        v[i] = sum / l[i * n + i];
    }
    v
}

fn spd_inverse(a: &[f64], n: usize) -> Vec<f64> {
    let l = cholesky(a, n);
    let mut inverse = vec![0.0; n * n];
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        // L L' x = e: forward, then backward through L'.
        let v = forward_substitute(&l, &e, n);
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = v[i];
            for k in i + 1..n {
                sum -= l[k * n + i] * x[k];
            }
            x[i] = sum / l[i * n + i];
        }
        for row in 0..n {
            inverse[row * n + col] = x[row];
        }
    }
    inverse
}

/// The shipped defaults (`SamplerConfig::default()`: NUTS, 4 chains, 500
/// warmup, 1000 draws, target acceptance 0.80, metric auto, tree depth 10),
/// apart from the seed, a single worker thread and no progress bar.
pub fn default_config(seed: u64) -> SamplerConfig {
    SamplerConfig {
        seed,
        num_threads: 1,
        show_progress: false,
        ..Default::default()
    }
}

/// Fit `target` through `path` at `config`.
pub fn fit_with(target: &GaussianTarget, path: Path, config: SamplerConfig) -> SampleResult {
    match path {
        Path::Custom => sample_target(target, config, None),
        Path::Graph => sample(target.graph(), config),
    }
    .expect("sampling failed")
}

/// Fit `target` through `path` at the shipped defaults.
pub fn fit(target: &GaussianTarget, path: Path, seed: u64) -> SampleResult {
    fit_with(target, path, default_config(seed))
}

/// Which part of the gate a fit failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Check {
    Divergences,
    BulkEss,
    Mean,
    Variance,
    SquaredNorm,
}

/// The gate's verdict on one fit, with everything a report needs.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub label: String,
    /// Every failed check with its message; empty when the fit passes.
    pub failures: Vec<(Check, String)>,
    /// Signed standardized errors per coordinate, in MCSEs.
    pub mean_z: Vec<f64>,
    pub variance_z: Vec<f64>,
    /// Standardized error of the mean squared norm of the whitened draws.
    pub squared_norm_z: f64,
    pub ess_bulk: Vec<f64>,
    pub max_r_hat: f64,
    pub divergences: usize,
    /// Leapfrog steps recorded in warmup and in draw transitions.
    pub warmup_leapfrog_steps: usize,
    pub draw_leapfrog_steps: usize,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn failed(&self, check: Check) -> bool {
        self.failures.iter().any(|(c, _)| *c == check)
    }

    /// Smallest bulk ESS over coordinates; NaN if any is NaN.
    pub fn min_ess_bulk(&self) -> f64 {
        worst(&self.ess_bulk, |e| -e).map_or(f64::NAN, |(_, e)| e)
    }

    pub fn worst_mean_z(&self) -> f64 {
        worst(&self.mean_z, f64::abs).map_or(f64::NAN, |(_, z)| z.abs())
    }

    pub fn worst_variance_z(&self) -> f64 {
        worst(&self.variance_z, f64::abs).map_or(f64::NAN, |(_, z)| z.abs())
    }

    /// One line for logs, printed before any assertion so a failing run
    /// still reports it.
    pub fn summary(&self) -> String {
        let total = self.warmup_leapfrog_steps + self.draw_leapfrog_steps;
        format!(
            "{}: worst mean error {:.2} MCSE, worst variance error {:.2} MCSE, squared norm \
             error {:.2} MCSE, min bulk ESS {:.0}, {} divergences, max R-hat {:.4}; leapfrog \
             steps per min bulk ESS {:.1} (draws only {:.1}){}",
            self.label,
            self.worst_mean_z(),
            self.worst_variance_z(),
            self.squared_norm_z,
            self.min_ess_bulk(),
            self.divergences,
            self.max_r_hat,
            total as f64 / self.min_ess_bulk(),
            self.draw_leapfrog_steps as f64 / self.min_ess_bulk(),
            if self.passed() {
                String::from("; PASS")
            } else {
                format!(
                    "; FAIL ({})",
                    self.failures
                        .iter()
                        .map(|(_, m)| m.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            }
        )
    }

    /// Panic with every failure unless the fit passed.
    pub fn assert_passed(&self) {
        assert!(self.passed(), "{}", self.summary());
    }
}

/// The index and value of the element with the largest `key`, where NaN
/// counts as the largest of all.
fn worst(values: &[f64], key: impl Fn(f64) -> f64) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in values.iter().enumerate() {
        let k = key(v);
        match best {
            Some((_, b)) if key(b).is_nan() => {}
            Some((_, b)) if !k.is_nan() && k <= key(b) => {}
            _ => best = Some((i, v)),
        }
    }
    best
}

/// Apply the gate to `fit` against the exact moments of `target`.
pub fn check(label: &str, fit: &SampleResult, target: &GaussianTarget) -> Outcome {
    let (means, sds) = (&target.means, &target.sds);
    let report = fit.diagnostics();
    let squared: Vec<Vec<Vec<f64>>> = fit
        .samples
        .iter()
        .map(|chain| {
            chain
                .iter()
                .map(|draw| {
                    draw.iter()
                        .enumerate()
                        .map(|(i, &x)| (x - means[i]).powi(2))
                        .collect()
                })
                .collect()
        })
        .collect();
    let squared_report = compute_diagnostics(
        &squared,
        &fit.param_names,
        &report.accept_rates,
        report.divergences,
    );

    let mean_z: Vec<f64> = (0..means.len())
        .map(|i| (report.params[i].mean - means[i]) / report.params[i].mcse_mean)
        .collect();
    let variance_z: Vec<f64> = (0..means.len())
        .map(|i| {
            (squared_report.params[i].mean - sds[i] * sds[i]) / squared_report.params[i].mcse_mean
        })
        .collect();
    let d = means.len();
    let squared_norms: Vec<Vec<Vec<f64>>> = fit
        .samples
        .iter()
        .map(|chain| {
            chain
                .iter()
                .map(|draw| {
                    let w = mat_vec(&target.a, draw, d);
                    vec![w.iter().zip(&target.c).map(|(w, c)| (w - c).powi(2)).sum()]
                })
                .collect()
        })
        .collect();
    let squared_norm_report = compute_diagnostics(
        &squared_norms,
        &[String::from("squared_norm")],
        &report.accept_rates,
        report.divergences,
    );
    let squared_norm = &squared_norm_report.params[0];
    let squared_norm_z = (squared_norm.mean - d as f64) / squared_norm.mcse_mean;
    let ess_bulk: Vec<f64> = report.params.iter().map(|p| p.ess_bulk).collect();
    let max_r_hat = report
        .params
        .iter()
        .map(|p| p.r_hat)
        .fold(
            f64::NEG_INFINITY,
            |a, b| if b.is_nan() { b } else { a.max(b) },
        );
    let (mut warmup_leapfrog_steps, mut draw_leapfrog_steps) = (0, 0);
    for transition in fit.transitions.iter().flatten() {
        if transition.is_warmup {
            warmup_leapfrog_steps += transition.num_leapfrog_steps;
        } else {
            draw_leapfrog_steps += transition.num_leapfrog_steps;
        }
    }

    let mut failures = Vec::new();
    if report.divergences != 0 {
        failures.push((
            Check::Divergences,
            format!("{} divergent transitions", report.divergences),
        ));
    }
    if let Some((i, e)) = worst(&ess_bulk, |e| -e) {
        if !(e >= MIN_ESS_BULK) {
            failures.push((
                Check::BulkEss,
                format!("coordinate {i} has bulk ESS {e:.1}, below {MIN_ESS_BULK}"),
            ));
        }
    }
    if let Some((i, z)) = worst(&mean_z, f64::abs) {
        if !(z.abs() < MAX_MCSE) {
            failures.push((
                Check::Mean,
                format!(
                    "coordinate {i} mean {} is {z:.2} MCSE (MCSE {}) from the exact {}",
                    report.params[i].mean, report.params[i].mcse_mean, means[i]
                ),
            ));
        }
    }
    if let Some((i, z)) = worst(&variance_z, f64::abs) {
        if !(z.abs() < MAX_MCSE) {
            failures.push((
                Check::Variance,
                format!(
                    "coordinate {i} variance {} is {z:.2} MCSE (MCSE {}) from the exact {}",
                    squared_report.params[i].mean,
                    squared_report.params[i].mcse_mean,
                    sds[i] * sds[i]
                ),
            ));
        }
    }

    if !(squared_norm_z.abs() < MAX_MCSE) {
        failures.push((
            Check::SquaredNorm,
            format!(
                "mean squared norm of the whitened draws {} is {squared_norm_z:.2} MCSE (MCSE {}) \
                 from the exact {d}",
                squared_norm.mean, squared_norm.mcse_mean
            ),
        ));
    }

    Outcome {
        label: label.to_string(),
        failures,
        mean_z,
        variance_z,
        squared_norm_z,
        ess_bulk,
        max_r_hat,
        divergences: report.divergences,
        warmup_leapfrog_steps,
        draw_leapfrog_steps,
    }
}

/// The targets of the gate, each with the paths it runs through and the seed
/// the gate uses for it.
pub fn gate_cases() -> Vec<(GaussianTarget, Path, u64)> {
    vec![
        (GaussianTarget::independent_normals(100), Path::Custom, 2101),
        (GaussianTarget::independent_normals(100), Path::Graph, 2111),
        (GaussianTarget::correlated(50, 0.95), Path::Custom, 2201),
        (GaussianTarget::correlated(50, 0.95), Path::Graph, 2211),
        (GaussianTarget::conjugate_regression(), Path::Graph, 2301),
    ]
}

/// A label naming the target, its dimension and the path.
pub fn label(target: &GaussianTarget, path: Path) -> String {
    format!(
        "{} ({} dimensions), {} path",
        target.name,
        target.dimension(),
        path.name()
    )
}

/// The family-wise false-alarm rate the moment checks alone would have for a
/// target of `dimension` coordinates if every standardized error were an
/// independent standard normal: two checks per coordinate and the squared
/// norm. (On the correlated targets the coordinates' errors are correlated,
/// which can only lower the true rate below this.)
pub fn nominal_false_alarm_rate(dimension: usize) -> f64 {
    1.0 - (1.0 - NOMINAL_EXCEEDANCE).powi(2 * dimension as i32 + 1)
}

/// The shipped defaults with `sampler` in place of NUTS.
pub fn with_sampler(seed: u64, sampler: SamplerType) -> SamplerConfig {
    SamplerConfig {
        sampler,
        ..default_config(seed)
    }
}
