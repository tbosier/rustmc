//! Per-fit triage: which fits `fit_warnings` flags, and how it reports a check
//! the sampler cannot make.
//!
//! The two reference posteriors are the textbook pair. Independent standard
//! normals are what NUTS samples best, so a correct rule raises nothing on
//! them at the shipped defaults. The centred parameterization of the eight
//! schools model (Rubin 1981; Gelman et al., Bayesian Data Analysis, 5.5) has
//! a funnel between the group scale and the school effects that NUTS cannot
//! follow at the shipped target acceptance of 0.8, which shows up as
//! divergences, and the rule must flag it on every seed.
use rustmc_core::bayesian_forecast::{fit_bayesian_local_level, BayesianLocalLevelConfig};
use rustmc_core::bayesian_trend::{
    fit_bayesian_local_linear_trend, local_linear_trend_limitation, BayesianLocalLinearTrendConfig,
};
use rustmc_core::diagnostics::{
    fit_warnings, FitWarning, SamplerTelemetry, WarningCode, WarningStatus,
};
use rustmc_core::forecast_common::InverseGammaPrior;
use rustmc_core::sampler::{SampleResult, SamplerConfig, SamplerType};
use rustmc_core::target::{sample_target, LogDensity};

const SEEDS: [u64; 3] = [1, 2, 3];

/// `dimension` independent standard normals.
struct StandardNormals(usize);

impl LogDensity for StandardNormals {
    fn dimension(&self) -> usize {
        self.0
    }
    fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
        let mut log_density = 0.0;
        for (g, x) in gradient.iter_mut().zip(q) {
            log_density -= 0.5 * x * x;
            *g = -x;
        }
        Ok(log_density)
    }
}

/// Centred eight schools: `mu ~ Normal(0, 5)`, `tau ~ HalfCauchy(5)` sampled
/// as `log tau`, `theta[j] ~ Normal(mu, tau)` and `y[j] ~ Normal(theta[j],
/// sigma[j])`, with the published estimates and standard errors.
struct CentredEightSchools;

const Y: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const SIGMA: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];

impl LogDensity for CentredEightSchools {
    fn dimension(&self) -> usize {
        10
    }
    fn parameter_names(&self) -> Vec<String> {
        let mut names = vec!["mu".to_string(), "log_tau".to_string()];
        names.extend((0..8).map(|j| format!("theta[{j}]")));
        names
    }
    fn log_density_gradient(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, String> {
        let (mu, log_tau) = (q[0], q[1]);
        let tau = log_tau.exp();
        // Normal(0, 5) on mu.
        let mut log_density = -0.5 * (mu / 5.0).powi(2);
        gradient[0] = -mu / 25.0;
        // HalfCauchy(5) on tau, plus the log Jacobian log tau.
        let ratio = tau / 5.0;
        log_density += -(1.0 + ratio * ratio).ln() + log_tau;
        gradient[1] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        for j in 0..8 {
            let theta = q[2 + j];
            let z = (theta - mu) / tau;
            log_density += -0.5 * z * z - log_tau;
            gradient[0] += z / tau;
            gradient[1] += z * z - 1.0;
            let residual = (Y[j] - theta) / SIGMA[j];
            log_density += -0.5 * residual * residual;
            gradient[2 + j] = -z / tau + residual / SIGMA[j];
        }
        Ok(log_density)
    }
}

/// The reference model's gradient matches central differences of its density,
/// so its divergences come from its geometry, not from a wrong gradient.
#[test]
fn centred_eight_schools_gradient_matches_finite_differences() {
    let model = CentredEightSchools;
    for point in [
        [1.0, 0.5, 3.0, -2.0, 0.0, 5.0, 1.0, -1.0, 2.0, 4.0],
        [-2.0, -1.5, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
    ] {
        let mut gradient = [0.0; 10];
        model.log_density_gradient(&point, &mut gradient).unwrap();
        for i in 0..10 {
            let (mut plus, mut minus) = (point, point);
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            let mut scratch = [0.0; 10];
            let difference = (model.log_density_gradient(&plus, &mut scratch).unwrap()
                - model.log_density_gradient(&minus, &mut scratch).unwrap())
                / 2e-6;
            assert!(
                (gradient[i] - difference).abs() < 1e-5 * difference.abs().max(1.0),
                "coordinate {i}: {} vs {difference}",
                gradient[i]
            );
        }
    }
}

/// Shipped defaults: NUTS, 4 chains, 500 warmup, 1000 draws, target
/// acceptance 0.8, maximum tree depth 10, the automatic metric.
fn defaults(seed: u64) -> SamplerConfig {
    SamplerConfig {
        seed,
        show_progress: false,
        ..SamplerConfig::default()
    }
}

fn hamiltonian_warnings(result: &SampleResult) -> Vec<FitWarning> {
    let transitions = result.transition_diagnostics();
    fit_warnings(
        &result.diagnostics(),
        SamplerTelemetry::Hamiltonian(&transitions),
    )
}

fn failures(warnings: &[FitWarning]) -> Vec<&FitWarning> {
    warnings
        .iter()
        .filter(|w| w.status == WarningStatus::Fail)
        .collect()
}

/// Every record states the threshold it was judged against.
fn assert_messages_state_thresholds(warnings: &[FitWarning]) {
    for warning in warnings {
        assert!(
            warning.message.contains(&format!("{}", warning.threshold)),
            "{:?} does not state its threshold {}: {}",
            warning.code,
            warning.threshold,
            warning.message
        );
    }
}

#[test]
fn independent_normals_raise_nothing_at_the_shipped_defaults() {
    for seed in SEEDS {
        let result = sample_target(&StandardNormals(10), defaults(seed), None).unwrap();
        let warnings = hamiltonian_warnings(&result);
        assert!(warnings.is_empty(), "seed {seed}: {warnings:#?}");
        let report = result.transition_diagnostics();
        for chain in &report.chains {
            assert!(chain.e_bfmi > 0.3, "seed {seed}: {chain:?}");
        }
        assert_eq!(report.total_tree_depth_saturation, Some(0));
    }
}

/// At stationarity the energy is `-log p(q) + p' M^-1 p / 2`, so for ten
/// standard normals with `log p = -|q|^2 / 2` its expectation is 10 whatever
/// metric warmup adapted. The energy's standard deviation is about
/// `sqrt(10)`, so the bound of 0.4 on the mean of 4000 autocorrelated draws
/// is loose; with this seed the mean is 9.88.
#[test]
fn nuts_energy_has_the_stationary_expectation() {
    let result = sample_target(&StandardNormals(10), defaults(5), None).unwrap();
    let energies: Vec<f64> = result
        .transitions
        .iter()
        .flatten()
        .filter(|t| !t.is_warmup)
        .map(|t| t.energy)
        .collect();
    assert_eq!(energies.len(), 4000);
    let mean = energies.iter().sum::<f64>() / energies.len() as f64;
    assert!((mean - 10.0).abs() < 0.4, "{mean}");
    // The energy is the position's potential plus a non-negative kinetic
    // energy, so it is never below the retained draw's potential.
    for (transitions, draws) in result.transitions.iter().zip(&result.samples) {
        let retained = transitions.iter().filter(|t| !t.is_warmup);
        for (transition, draw) in retained.zip(draws) {
            let potential = 0.5 * draw.iter().map(|x| x * x).sum::<f64>();
            assert!(transition.energy >= potential - 1e-9);
        }
    }
}

#[test]
fn centred_eight_schools_is_flagged_on_every_seed() {
    for seed in SEEDS {
        let result = sample_target(&CentredEightSchools, defaults(seed), None).unwrap();
        let warnings = hamiltonian_warnings(&result);
        let failed = failures(&warnings);
        // Divergences and a low E-BFMI are the funnel's signature; on these
        // seeds R-hat and ESS fail as well.
        for code in [WarningCode::Divergences, WarningCode::EBfmi] {
            assert!(
                failed.iter().any(|w| w.code == code),
                "seed {seed}, {code:?}: {warnings:#?}"
            );
        }
        assert_messages_state_thresholds(&warnings);
    }
}

#[test]
fn hmc_reports_tree_depth_saturation_once_as_unavailable() {
    let config = SamplerConfig {
        sampler: SamplerType::Hmc,
        // Ten steps rather than the default fifteen, which resonates on a
        // standard normal (see analytic_posteriors.rs).
        num_leapfrog_steps: 10,
        ..defaults(4)
    };
    let result = sample_target(&StandardNormals(4), config, None).unwrap();
    let warnings = hamiltonian_warnings(&result);
    let saturation: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::TreeDepthSaturation)
        .collect();
    assert_eq!(saturation.len(), 1, "{warnings:#?}");
    assert_eq!(saturation[0].status, WarningStatus::Unavailable);
    assert!(saturation[0].message.contains("static HMC"));
    // HMC has an energy, so E-BFMI is checked rather than unavailable.
    assert!(!warnings.iter().any(|w| w.code == WarningCode::EBfmi));
    assert_messages_state_thresholds(&warnings);
}

fn gibbs_series() -> Vec<f64> {
    (0..80)
        .map(|t| (t as f64 * 0.3).sin() + 0.05 * t as f64)
        .collect()
}

#[test]
fn gibbs_fits_report_hamiltonian_checks_as_unavailable_never_as_passes() {
    let config = BayesianLocalLevelConfig {
        initial_mean: 0.0,
        initial_variance: 10.0,
        process_variance_prior: InverseGammaPrior::new(3.0, 0.5).unwrap(),
        observation_variance_prior: InverseGammaPrior::new(3.0, 0.5).unwrap(),
        num_chains: 4,
        num_warmup: 200,
        num_draws: 300,
        thinning: 1,
        seed: 5,
    };
    let posterior = fit_bayesian_local_level(&gibbs_series(), &config).unwrap();
    let warnings = fit_warnings(
        &posterior.diagnostics(),
        SamplerTelemetry::NonHamiltonian {
            sampler: "conjugate Gibbs/FFBS",
        },
    );
    for code in [
        WarningCode::Divergences,
        WarningCode::EBfmi,
        WarningCode::TreeDepthSaturation,
    ] {
        let records: Vec<_> = warnings.iter().filter(|w| w.code == code).collect();
        assert_eq!(records.len(), 1, "{code:?}: {warnings:#?}");
        assert_eq!(records[0].status, WarningStatus::Unavailable);
        assert!(records[0].value.is_nan());
        assert!(records[0].message.contains("conjugate Gibbs/FFBS"));
    }
    assert_messages_state_thresholds(&warnings);
}

#[test]
fn too_few_draws_fail_and_a_fixed_parameter_is_only_unavailable() {
    let config = SamplerConfig {
        num_draws: 3,
        num_warmup: 50,
        ..defaults(6)
    };
    let result = sample_target(&StandardNormals(2), config, None).unwrap();
    let warnings = hamiltonian_warnings(&result);
    let unavailable: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::DiagnosticsUnavailable)
        .collect();
    assert_eq!(unavailable.len(), 2, "{warnings:#?}");
    for warning in unavailable {
        assert_eq!(warning.status, WarningStatus::Fail);
        assert_eq!(warning.value, 3.0);
        assert_eq!(warning.threshold, 4.0);
    }
    assert_messages_state_thresholds(&warnings);

    // A variance fixed by the model is constant in every chain: its R-hat
    // and ESS do not exist, which is not a failure of the fit.
    let mut fixed = result.diagnostics();
    fixed.num_draws = 100;
    for parameter in &mut fixed.params {
        parameter.mean = 2.0;
        parameter.r_hat = f64::NAN;
        parameter.ess_bulk = f64::NAN;
        parameter.ess_tail = f64::NAN;
    }
    let warnings = fit_warnings(&fixed, SamplerTelemetry::NonHamiltonian { sampler: "test" });
    let unavailable: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::DiagnosticsUnavailable)
        .collect();
    assert_eq!(unavailable.len(), 2);
    assert!(unavailable
        .iter()
        .all(|w| w.status == WarningStatus::Unavailable));
    assert!(failures(&warnings).is_empty(), "{warnings:#?}");
}

#[test]
fn every_local_linear_trend_fit_carries_its_known_limitation() {
    let config = BayesianLocalLinearTrendConfig {
        initial_mean: [0.0, 0.0],
        initial_covariance: [4.0, 0.0, 0.0, 1.0],
        level_variance_prior: InverseGammaPrior::new(3.0, 0.24).unwrap(),
        slope_variance_prior: InverseGammaPrior::new(3.0, 0.04).unwrap(),
        observation_variance_prior: InverseGammaPrior::new(3.0, 0.7).unwrap(),
        num_chains: 2,
        num_warmup: 50,
        num_draws: 60,
        thinning: 1,
        seed: 8,
    };
    let posterior = fit_bayesian_local_linear_trend(&gibbs_series(), &config).unwrap();
    let warnings = posterior.warnings();
    let limitation: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::KnownLimitation)
        .collect();
    let expected = local_linear_trend_limitation();
    assert_eq!(limitation.len(), 1, "{warnings:#?}");
    assert_eq!(limitation[0].message, expected.message);
    assert_eq!(limitation[0].threshold, 1.01);
    assert_eq!(limitation[0].status, WarningStatus::Fail);
    assert!(limitation[0].message.contains("docs/reference.md"));
    assert_messages_state_thresholds(&warnings);
}

#[test]
fn codes_round_trip_through_their_names() {
    for code in WarningCode::ALL {
        assert_eq!(WarningCode::parse(code.as_str()), Some(code));
    }
    assert_eq!(WarningCode::parse("rhat"), None);
}
