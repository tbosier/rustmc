//! Sampler exactness on targets whose moments are known in closed form.
//!
//! The recovery suite compares posterior means with the parameters that
//! generated the data, under hand-picked tolerances. That cannot see a small
//! sampler bias: the data-generating value is not the posterior mean, and a
//! tolerance wide enough for that gap also hides a bias of a few tenths of a
//! posterior standard deviation. Here the target *is* the posterior, so the
//! only error left is Monte Carlo error, and every check is scaled by the
//! sampler's own Monte Carlo standard error (MCSE) for that coordinate.
//!
//! Every run uses the shipped defaults (`SamplerConfig::default()`: 4 chains,
//! 500 warmup, 1000 draws, target acceptance 0.80, 15 leapfrog steps for HMC)
//! and is checked, per coordinate:
//!
//! - the bulk ESS is at least 400, the release gate's floor, so that the
//!   MCSE below is small and estimated from enough draws to trust;
//! - the mean: `|mean - exact mean| < 4 * mcse_mean`;
//! - the variance: the draws of `(x - exact mean)^2` have expectation exactly
//!   the exact variance, so their mean is held to the same 4-MCSE bound,
//!   with an MCSE estimated from those squared draws' own autocorrelation.
//!
//! NUTS passes all of it. Static HMC does not, and its two cases are ignored
//! reproducers rather than checks loosened until they pass. On these
//! well-conditioned targets the adapted step size puts 15 leapfrog steps at an
//! integration time of 5.4 to 6.2 per chain (seed 2102), close to the 2*pi
//! period of a standardized Gaussian coordinate, so a transition returns
//! almost to where it started: from 4000 draws the worst coordinate's bulk
//! ESS is 7, and the median 170. With 10 steps the same run's worst bulk ESS
//! is 2164. The standardized errors of the coordinates that do mix were also
//! wider than a correct MCSE allows: on the correlated target, over six
//! seeds, their spread reached 2.8 times the expected one even at 10,000
//! draws.
//!
//! Each run also prints the leapfrog steps spent per unit of bulk ESS (the
//! smallest across coordinates), so a change to tree building or adaptation
//! can be measured, not only pass or fail.
use rustmc_core::diagnostics::compute_diagnostics;
use rustmc_core::sampler::{SampleResult, SamplerConfig, SamplerType};
use rustmc_core::target::{sample_target, LogDensity};

/// Bound on every standardized error, in MCSEs. A correct sampler with a
/// correct MCSE crosses it on about one check in sixteen thousand.
const MAX_MCSE: f64 = 4.0;

/// Minimum bulk ESS for every coordinate, as in the release gate.
const MIN_ESS_BULK: f64 = 400.0;

/// Independent normals with distinct means and scales, so a sampler that
/// mixes up coordinates, or adapts one shared scale, is caught.
struct IndependentNormals {
    means: Vec<f64>,
    sds: Vec<f64>,
}

impl IndependentNormals {
    fn new(dimension: usize) -> Self {
        let means = (0..dimension)
            .map(|i| (i as f64 * 0.37).sin() * 3.0)
            .collect();
        // Scales from 0.5 to 2.0: a four-fold spread the metric must adapt to.
        let sds = (0..dimension)
            .map(|i| 0.5 + 1.5 * i as f64 / (dimension - 1) as f64)
            .collect();
        Self { means, sds }
    }
}

impl LogDensity for IndependentNormals {
    fn dimension(&self) -> usize {
        self.means.len()
    }

    fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
        let mut log_density = 0.0;
        for i in 0..q.len() {
            let z = (q[i] - self.means[i]) / self.sds[i];
            log_density -= 0.5 * z * z;
            gradient[i] = -z / self.sds[i];
        }
        Ok(log_density)
    }
}

/// A Gaussian whose correlation matrix is `rho^|i - j|` (a stationary AR(1)
/// correlation) with per-coordinate scales. Its precision is tridiagonal, so
/// the density and gradient are exact and cheap; the marginal variance of
/// coordinate `i` is `sds[i]^2`. A diagonal metric cannot remove this
/// correlation, so the sampler has to cope with it.
struct CorrelatedGaussian {
    means: Vec<f64>,
    sds: Vec<f64>,
    rho: f64,
}

impl CorrelatedGaussian {
    fn new(dimension: usize, rho: f64) -> Self {
        let means = (0..dimension).map(|i| 1.0 - 0.05 * i as f64).collect();
        let sds = (0..dimension).map(|i| 1.0 + 0.5 * (i % 3) as f64).collect();
        Self { means, sds, rho }
    }
}

impl LogDensity for CorrelatedGaussian {
    fn dimension(&self) -> usize {
        self.means.len()
    }

    fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
        // With z = D^-1 (q - mean), the AR(1) correlation inverse is
        // (1 / (1 - rho^2)) * tridiag(-rho, [1, 1 + rho^2, ..., 1 + rho^2, 1], -rho).
        let n = q.len();
        let rho = self.rho;
        let z: Vec<f64> = (0..n)
            .map(|i| (q[i] - self.means[i]) / self.sds[i])
            .collect();
        let scale = 1.0 / (1.0 - rho * rho);
        let mut quadratic = 0.0;
        for i in 0..n {
            let diagonal = if i == 0 || i == n - 1 {
                1.0
            } else {
                1.0 + rho * rho
            };
            let mut precision_z = diagonal * z[i];
            if i > 0 {
                precision_z -= rho * z[i - 1];
            }
            if i + 1 < n {
                precision_z -= rho * z[i + 1];
            }
            precision_z *= scale;
            quadratic += z[i] * precision_z;
            gradient[i] = -precision_z / self.sds[i];
        }
        Ok(-0.5 * quadratic)
    }
}

/// The shipped defaults, apart from the seed and the progress bar.
fn default_config(sampler: SamplerType, seed: u64) -> SamplerConfig {
    SamplerConfig {
        sampler,
        seed,
        num_threads: 1,
        show_progress: false,
        ..Default::default()
    }
}

/// Draws of `f(chain, coordinate, value)` in the `[chain][draw][param]` layout
/// the diagnostics take.
fn map_draws(fit: &SampleResult, f: impl Fn(usize, f64) -> f64) -> Vec<Vec<Vec<f64>>> {
    fit.samples
        .iter()
        .map(|chain| {
            chain
                .iter()
                .map(|draw| draw.iter().enumerate().map(|(i, &x)| f(i, x)).collect())
                .collect()
        })
        .collect()
}

/// Whether `candidate` is worse than `current` when larger is worse and NaN is
/// worst of all.
fn worse(candidate: f64, current: f64) -> bool {
    !current.is_nan() && (candidate.is_nan() || candidate > current)
}

fn check_moments(label: &str, fit: &SampleResult, means: &[f64], sds: &[f64]) {
    let report = fit.diagnostics();
    let squared = map_draws(fit, |i, x| (x - means[i]).powi(2));
    let squared_report = compute_diagnostics(
        &squared,
        &fit.param_names,
        &report.accept_rates,
        report.divergences,
    );

    let (mut worst_mean, mut worst_variance) = ((0.0_f64, 0), (0.0_f64, 0));
    let mut weakest_ess = (f64::INFINITY, 0);
    for i in 0..means.len() {
        let (param, squared_param) = (&report.params[i], &squared_report.params[i]);
        // A NaN diagnostic counts as the worst value and stays the worst, so
        // the assertions below catch it.
        if worse(-param.ess_bulk, -weakest_ess.0) {
            weakest_ess = (param.ess_bulk, i);
        }
        let mean_error = (param.mean - means[i]).abs() / param.mcse_mean;
        if worse(mean_error, worst_mean.0) {
            worst_mean = (mean_error, i);
        }
        let variance_error = (squared_param.mean - sds[i] * sds[i]).abs() / squared_param.mcse_mean;
        if worse(variance_error, worst_variance.0) {
            worst_variance = (variance_error, i);
        }
    }
    let draw_leapfrog_steps: usize = fit
        .transitions
        .iter()
        .flatten()
        .filter(|transition| !transition.is_warmup)
        .map(|transition| transition.num_leapfrog_steps)
        .sum();
    // Printed before the assertions, so a failing run still reports it.
    println!(
        "{label}: worst mean error {:.2} MCSE, worst variance error {:.2} MCSE, \
         {:.1} leapfrog steps per bulk ESS (min bulk ESS {:.0})",
        worst_mean.0,
        worst_variance.0,
        draw_leapfrog_steps as f64 / weakest_ess.0,
        weakest_ess.0
    );

    assert_eq!(report.divergences, 0, "{label}: divergent transitions");
    assert!(
        weakest_ess.0 >= MIN_ESS_BULK,
        "{label}: coordinate {} has bulk ESS {:.1}, below {MIN_ESS_BULK}",
        weakest_ess.1,
        weakest_ess.0
    );
    assert!(
        worst_mean.0 < MAX_MCSE,
        "{label}: coordinate {} mean {} is {:.2} MCSE (MCSE {}) from the exact {}",
        worst_mean.1,
        report.params[worst_mean.1].mean,
        worst_mean.0,
        report.params[worst_mean.1].mcse_mean,
        means[worst_mean.1]
    );
    assert!(
        worst_variance.0 < MAX_MCSE,
        "{label}: coordinate {} variance {} is {:.2} MCSE (MCSE {}) from the exact {}",
        worst_variance.1,
        squared_report.params[worst_variance.1].mean,
        worst_variance.0,
        squared_report.params[worst_variance.1].mcse_mean,
        sds[worst_variance.1].powi(2)
    );
}

fn run(label: &str, sampler: SamplerType, seed: u64, target: &impl Moments) {
    let fit = sample_target(target, default_config(sampler, seed), None).expect("sampling failed");
    check_moments(label, &fit, target.means(), target.sds());
}

/// The exact moments a target's draws are checked against.
trait Moments: LogDensity {
    fn means(&self) -> &[f64];
    fn sds(&self) -> &[f64];
}

impl Moments for IndependentNormals {
    fn means(&self) -> &[f64] {
        &self.means
    }
    fn sds(&self) -> &[f64] {
        &self.sds
    }
}

impl Moments for CorrelatedGaussian {
    fn means(&self) -> &[f64] {
        &self.means
    }
    fn sds(&self) -> &[f64] {
        &self.sds
    }
}

#[test]
fn nuts_matches_the_exact_moments_of_a_hundred_independent_normals() {
    run(
        "NUTS, 100 independent normals",
        SamplerType::Nuts,
        2101,
        &IndependentNormals::new(100),
    );
}

#[test]
fn nuts_matches_the_exact_moments_of_a_correlated_fifty_dimensional_gaussian() {
    run(
        "NUTS, 50 correlated Gaussian coordinates",
        SamplerType::Nuts,
        2201,
        &CorrelatedGaussian::new(50, 0.8),
    );
}

/// Reproducer for an open finding (see the module documentation): at the
/// default 15 leapfrog steps, static HMC's integration time lands near a full
/// period of these Gaussians and some coordinates barely move.
#[test]
#[ignore = "open finding: default static HMC nearly resonates on well-conditioned Gaussians"]
fn hmc_matches_the_exact_moments_of_a_hundred_independent_normals() {
    run(
        "HMC, 100 independent normals",
        SamplerType::Hmc,
        2102,
        &IndependentNormals::new(100),
    );
}

/// Reproducer for the same open finding on the correlated target, where the
/// coordinates that do mix also have errors wider than their MCSE allows.
#[test]
#[ignore = "open finding: default static HMC nearly resonates on well-conditioned Gaussians"]
fn hmc_matches_the_exact_moments_of_a_correlated_fifty_dimensional_gaussian() {
    run(
        "HMC, 50 correlated Gaussian coordinates",
        SamplerType::Hmc,
        2202,
        &CorrelatedGaussian::new(50, 0.8),
    );
}
