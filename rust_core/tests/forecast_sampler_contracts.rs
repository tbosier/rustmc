//! Size limits and data requirements the Gaussian forecasting samplers share.
//!
//! A Rust allocation that fails aborts the process, so an oversized horizon or
//! draw count must be refused with an error before anything is reserved. These
//! tests ask for sizes that would otherwise abort (or run for hours) and expect
//! a typed configuration error instead.

use rustmc_core::bayesian_ar::{
    fit_bayesian_ar, BayesianArConfig, BayesianArPosterior, BayesianArPosteriorDraw,
    NormalInverseGammaPrior,
};
use rustmc_core::bayesian_forecast::{
    fit_bayesian_local_level, BayesianForecastError, BayesianLocalLevelConfig, InverseGammaPrior,
    LocalLevelPosterior, LocalLevelPosteriorDraw,
};
use rustmc_core::bayesian_regression::{
    fit_regression, GaussianCoefficientPrior, RegressionConfig,
};
use rustmc_core::bayesian_seasonal::{
    fit_bayesian_seasonal_local_level, BayesianSeasonalLocalLevelConfig,
    SeasonalLocalLevelPosterior, SeasonalLocalLevelPosteriorDraw,
};
use rustmc_core::bayesian_trend::{
    fit_bayesian_local_linear_trend, BayesianLocalLinearTrendConfig, LocalLinearTrendPosterior,
    LocalLinearTrendPosteriorDraw,
};
use rustmc_core::forecast_common::{checked_value_count, MAX_MATERIALIZED_VALUES};
use rustmc_core::hierarchical::{HierarchicalMeanPosterior, HierarchicalMeanPosteriorDraw};
use rustmc_core::hurdle::{HurdleLogNormalDraw, HurdleLogNormalPosterior};
use rustmc_core::structural::{
    fit, Component, SamplingConfig, StructuralConfig, VarianceParameter,
};

/// Four chains of a thousand draws, the Python default.
const CHAINS: usize = 4;
const DRAWS: usize = 1000;
/// A horizon whose paths alone would need gigabytes.
const HUGE_HORIZON: usize = 100_000_000;

fn ig(shape: f64, scale: f64) -> InverseGammaPrior {
    InverseGammaPrior::new(shape, scale).unwrap()
}

fn assert_size_refusal(result: Result<impl std::fmt::Debug, BayesianForecastError>) {
    match result {
        Err(BayesianForecastError::InvalidConfiguration(message)) => {
            assert!(message.contains("safety limit"), "{message}");
        }
        other => panic!("expected a size refusal, got {other:?}"),
    }
}

fn local_level_config() -> BayesianLocalLevelConfig {
    BayesianLocalLevelConfig {
        initial_mean: 0.0,
        initial_variance: 4.0,
        process_variance_prior: ig(3.0, 0.2),
        observation_variance_prior: ig(3.0, 0.4),
        num_chains: 2,
        num_warmup: 5,
        num_draws: 5,
        thinning: 1,
        seed: 1,
    }
}

fn trend_config() -> BayesianLocalLinearTrendConfig {
    BayesianLocalLinearTrendConfig {
        initial_mean: [0.0, 0.0],
        initial_covariance: [4.0, 0.0, 0.0, 1.0],
        level_variance_prior: ig(3.0, 0.2),
        slope_variance_prior: ig(3.0, 0.02),
        observation_variance_prior: ig(3.0, 0.4),
        num_chains: 2,
        num_warmup: 5,
        num_draws: 5,
        thinning: 1,
        seed: 2,
    }
}

fn seasonal_config() -> BayesianSeasonalLocalLevelConfig {
    BayesianSeasonalLocalLevelConfig {
        period: 4,
        initial_level: 0.0,
        initial_seasonal_effects: vec![0.0; 4],
        initial_level_variance: 4.0,
        initial_seasonal_variance: 1.0,
        level_variance_prior: ig(3.0, 0.2),
        seasonal_variance_prior: ig(3.0, 0.1),
        observation_variance_prior: ig(3.0, 0.4),
        num_chains: 2,
        num_warmup: 5,
        num_draws: 5,
        thinning: 1,
        seed: 3,
    }
}

#[test]
fn the_shared_limit_is_the_one_every_model_reports() {
    assert_eq!(MAX_MATERIALIZED_VALUES, 25_000_000);
    let error =
        checked_value_count("probe", &[CHAINS, DRAWS, HUGE_HORIZON], 25_000_000).unwrap_err();
    assert_eq!(error.requested, Some(400_000_000_000));
    assert!(error.to_string().contains("25 million"));
}

#[test]
fn oversized_forecasts_are_refused_before_any_path_is_allocated() {
    let local = LocalLevelPosterior {
        chains: vec![
            vec![
                LocalLevelPosteriorDraw {
                    process_variance: 0.1,
                    observation_variance: 0.2,
                    terminal_level: 0.0,
                };
                DRAWS
            ];
            CHAINS
        ],
    };
    assert_size_refusal(local.forecast(HUGE_HORIZON, 1));
    assert_size_refusal(local.forecast(usize::MAX, 1));

    let trend = LocalLinearTrendPosterior {
        chains: vec![
            vec![
                LocalLinearTrendPosteriorDraw {
                    level_variance: 0.1,
                    slope_variance: 0.01,
                    observation_variance: 0.2,
                    terminal_level: 0.0,
                    terminal_slope: 0.0,
                };
                DRAWS
            ];
            CHAINS
        ],
    };
    assert_size_refusal(trend.forecast(HUGE_HORIZON, 1));

    let seasonal = SeasonalLocalLevelPosterior {
        period: 4,
        chains: vec![
            vec![
                SeasonalLocalLevelPosteriorDraw {
                    level_variance: 0.1,
                    seasonal_variance: 0.1,
                    observation_variance: 0.2,
                    terminal_state: vec![0.0; 4],
                };
                DRAWS
            ];
            CHAINS
        ],
    };
    assert_size_refusal(seasonal.forecast(HUGE_HORIZON, 1));

    let ar = BayesianArPosterior {
        order: 1,
        terminal_observations: vec![0.0],
        chains: vec![
            vec![
                BayesianArPosteriorDraw {
                    coefficients: vec![0.0, 0.5],
                    innovation_variance: 1.0,
                };
                DRAWS
            ];
            CHAINS
        ],
    };
    assert_size_refusal(ar.forecast(HUGE_HORIZON, 1));

    let hurdle = HurdleLogNormalPosterior {
        chains: vec![
            vec![
                HurdleLogNormalDraw {
                    payment_probability: 0.5,
                    process_variance: 0.01,
                    observation_variance: 0.1,
                    terminal_log_level: 0.0,
                };
                DRAWS
            ];
            CHAINS
        ],
        time_count: 1,
        observed_count: 1,
        positive_count: 1,
    };
    assert_size_refusal(hurdle.forecast(HUGE_HORIZON, 1));

    let hierarchical = HierarchicalMeanPosterior {
        chains: vec![
            vec![
                HierarchicalMeanPosteriorDraw {
                    population_mean: 0.0,
                    group_variance: 1.0,
                    program_variance: 1.0,
                    observation_variance: 1.0,
                    group_means: vec![0.0],
                    program_means: vec![0.0],
                };
                DRAWS
            ];
            CHAINS
        ],
        group_index: vec![0],
        observed_counts: vec![1],
        group_count: 1,
    };
    assert_size_refusal(hierarchical.forecast(HUGE_HORIZON, 1));
}

#[test]
fn a_regression_forecast_is_limited_by_its_design_rows() {
    let config = RegressionConfig::from_local_level(
        &local_level_config(),
        GaussianCoefficientPrior {
            mean: vec![0.0],
            covariance: vec![1.0],
        },
    )
    .unwrap();
    let y = [0.1, 0.4, -0.2, 0.3];
    let x: Vec<Vec<f64>> = y.iter().map(|_| vec![1.0]).collect();
    let posterior = fit_regression(&y, &x, &config).unwrap();
    // 2 chains x 5 draws x 6 path arrays: 420k rows is just over the limit.
    let rows = vec![vec![1.0]; 420_000];
    let message = posterior.forecast(&rows, 1).unwrap_err().to_string();
    assert!(message.contains("safety limit"), "{message}");
    assert!(posterior.forecast(&rows[..10], 1).is_ok());
}

#[test]
fn oversized_fits_are_refused_instead_of_aborting() {
    // Vec::with_capacity(num_draws) used to run before any size check, so a
    // draw count like this one aborted on a capacity overflow.
    let draws = usize::MAX / 8;
    let observations = [0.1, 0.4, -0.2, 0.3, 0.5];

    let mut config = local_level_config();
    config.num_warmup = 0;
    config.num_draws = draws;
    assert_size_refusal(fit_bayesian_local_level(&observations, &config));

    let mut config = trend_config();
    config.num_warmup = 0;
    config.num_draws = draws;
    assert_size_refusal(fit_bayesian_local_linear_trend(&observations, &config));

    let mut config = seasonal_config();
    config.num_warmup = 0;
    config.num_draws = draws;
    assert_size_refusal(fit_bayesian_seasonal_local_level(&observations, &config));

    let mut config = seasonal_config();
    config.period = 4000;
    config.initial_seasonal_effects = vec![0.0; 4000];
    // One chain's FFBS working state alone: 6 x 4000^2 x 3 values.
    assert_size_refusal(fit_bayesian_seasonal_local_level(&observations, &config));

    let ar = BayesianArConfig {
        order: 1,
        prior: NormalInverseGammaPrior::new(
            vec![0.0; 2],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            3.0,
            1.0,
        )
        .unwrap(),
        num_chains: 4,
        num_draws: draws,
        seed: 1,
    };
    assert_size_refusal(fit_bayesian_ar(&observations, &ar));

    let regression = {
        let mut base = local_level_config();
        base.num_warmup = 0;
        base.num_draws = draws;
        RegressionConfig::from_local_level(
            &base,
            GaussianCoefficientPrior {
                mean: vec![0.0],
                covariance: vec![1.0],
            },
        )
        .unwrap()
    };
    let x: Vec<Vec<f64>> = observations.iter().map(|_| vec![1.0]).collect();
    let message = fit_regression(&observations, &x, &regression)
        .unwrap_err()
        .to_string();
    assert!(message.contains("safety limit"), "{message}");
}

#[test]
fn each_fit_needs_one_finite_observation_per_inferred_variance() {
    let two = [0.1, f64::NAN, 0.3, f64::NAN];
    let three = [0.1, f64::NAN, 0.3, 0.2];

    // Process and observation variances.
    assert!(fit_bayesian_local_level(&[0.1, f64::NAN], &local_level_config()).is_err());
    assert!(fit_bayesian_local_level(&two, &local_level_config()).is_ok());

    // Level, slope and observation variances.
    let error = fit_bayesian_local_linear_trend(&two, &trend_config()).unwrap_err();
    assert!(matches!(
        error,
        BayesianForecastError::InvalidObservations(_)
    ));
    assert!(error.to_string().contains("at least 3"), "{error}");
    assert!(fit_bayesian_local_linear_trend(&three, &trend_config()).is_ok());

    // Level, seasonal and observation variances: two used to be accepted.
    let error = fit_bayesian_seasonal_local_level(&two, &seasonal_config()).unwrap_err();
    assert!(matches!(
        error,
        BayesianForecastError::InvalidObservations(_)
    ));
    assert!(fit_bayesian_seasonal_local_level(&three, &seasonal_config()).is_ok());

    // A trend regression infers the same three variances as the trend.
    let regression = RegressionConfig::from_trend(
        &trend_config(),
        GaussianCoefficientPrior {
            mean: vec![0.0],
            covariance: vec![1.0],
        },
    )
    .unwrap();
    let x = vec![vec![1.0]; 4];
    assert!(fit_regression(&two, &x, &regression).is_err());
    assert!(fit_regression(&three, &x, &regression).is_ok());

    // Structural: count the inverse-gamma variances, and at least one.
    let sampling = SamplingConfig {
        chains: 1,
        draws: 3,
        warmup: 1,
        thinning: 1,
        seed: 4,
        store_states: false,
    };
    let fixed = StructuralConfig {
        components: vec![Component::level(
            "level".into(),
            VarianceParameter::Fixed(0.1),
            0.0,
            1.0,
        )],
        observation_variance: VarianceParameter::Fixed(0.2),
        student_df: None,
    };
    assert!(fit(&[f64::NAN, f64::NAN], None, &fixed, &sampling).is_err());
    assert!(fit(&[f64::NAN, 0.3], None, &fixed, &sampling).is_ok());
    let mut inferred = fixed;
    inferred.components[0].innovations[0] = VarianceParameter::InverseGamma {
        shape: 3.0,
        scale: 0.2,
    };
    inferred.observation_variance = VarianceParameter::InverseGamma {
        shape: 3.0,
        scale: 0.4,
    };
    assert!(fit(&[f64::NAN, 0.3], None, &inferred, &sampling).is_err());
    assert!(fit(&[0.1, 0.3], None, &inferred, &sampling).is_ok());
}
