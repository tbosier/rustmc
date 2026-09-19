//! Numerical-stability regressions for the specialised Gibbs/FFBS forecasting
//! models.
//!
//! Every expectation here is a closed form or a constant produced by an
//! out-of-crate high-precision evaluation: the recursion is replayed over the
//! exact rational values of the stored doubles with Python's `fractions`, then
//! rounded once to `f64` through `decimal` at 400 significant digits. Nothing
//! in this file compares one in-repo implementation against another, because
//! the four forecasting modules are copies of one another and would agree on a
//! wrong formula.

use rustmc_core::bayesian_ar::BayesianArForecast;
use rustmc_core::bayesian_forecast::{
    fit_bayesian_local_level, BayesianLocalLevelConfig, InverseGammaPrior,
    PosteriorPredictiveForecast,
};
use rustmc_core::bayesian_seasonal::SeasonalPosteriorPredictiveForecast;
use rustmc_core::bayesian_trend::TrendPosteriorPredictiveForecast;

// ---------------------------------------------------------------------------
// Defect 1 — forecast mean accessors overflow on finite inputs
// ---------------------------------------------------------------------------

/// `[chain][draw][step]` from one column of draw values per horizon step.
fn chains(columns: &[&[f64]]) -> Vec<Vec<Vec<f64>>> {
    let n_draws = columns[0].len();
    assert!(columns.iter().all(|column| column.len() == n_draws));
    let draws: Vec<Vec<f64>> = (0..n_draws)
        .map(|draw| columns.iter().map(|column| column[draw]).collect())
        .collect();
    // Two chains, so the accessors are exercised across the chain axis rather
    // than over one flat list.
    let split = n_draws / 2;
    vec![draws[..split].to_vec(), draws[split..].to_vec()]
}

/// Every per-horizon mean the four forecast results expose, so a fix applied to
/// one copy of `path_means` and not the others is caught.
fn all_means(columns: &[&[f64]]) -> Vec<(&'static str, Vec<f64>)> {
    let paths = chains(columns);
    let local_level = PosteriorPredictiveForecast {
        state_paths: paths.clone(),
        observation_paths: paths.clone(),
    };
    let trend = TrendPosteriorPredictiveForecast {
        level_paths: paths.clone(),
        slope_paths: paths.clone(),
        observation_paths: paths.clone(),
    };
    let seasonal = SeasonalPosteriorPredictiveForecast {
        level_paths: paths.clone(),
        seasonal_paths: paths.clone(),
        observation_paths: paths.clone(),
        cumulative_observation_paths: paths.clone(),
    };
    let ar = BayesianArForecast {
        conditional_mean_paths: paths.clone(),
        observation_paths: paths,
    };
    vec![
        (
            "local_level.state_means",
            local_level.state_means().unwrap(),
        ),
        (
            "local_level.observation_means",
            local_level.observation_means().unwrap(),
        ),
        ("trend.level_means", trend.level_means().unwrap()),
        ("trend.slope_means", trend.slope_means().unwrap()),
        (
            "trend.observation_means",
            trend.observation_means().unwrap(),
        ),
        ("seasonal.level_means", seasonal.level_means().unwrap()),
        (
            "seasonal.seasonal_means",
            seasonal.seasonal_means().unwrap(),
        ),
        (
            "seasonal.observation_means",
            seasonal.observation_means().unwrap(),
        ),
        (
            "seasonal.cumulative_observation_means",
            seasonal.cumulative_observation_means().unwrap(),
        ),
        (
            "ar.conditional_mean_means",
            ar.conditional_mean_means().unwrap(),
        ),
        ("ar.observation_means", ar.observation_means().unwrap()),
    ]
}

/// Draws near the top of the representable range have a representable mean.
/// Accumulating them and dividing at the end does not: `1e308 + 1e308` is
/// already infinite, and the infinity survives the division.
///
/// Step 0 is the mean of `[1e308, 1e308, -1e308, 1e308]`, whose exact value
/// rounds to `5e307`; step 1 is `[1e308, 1e308, -1e308, -1e308]`, exactly `0`;
/// step 2 is `[1e308, 1e308, 1e308, 1e308]`, exactly `1e308`.
#[test]
fn forecast_means_stay_finite_when_the_draws_sum_past_the_range() {
    let columns: [&[f64]; 3] = [
        &[1e308, 1e308, -1e308, 1e308],
        &[1e308, 1e308, -1e308, -1e308],
        &[1e308, 1e308, 1e308, 1e308],
    ];
    for (name, means) in all_means(&columns) {
        assert_eq!(means, vec![5e307, 0.0, 1e308], "{name}");
    }
}

/// The same accessors at an ordinary scale, where a running sum rounds once per
/// draw and the scaled form does not: the exact mean of `[0.1, 0.2, 0.3]` is
/// `0.2` to the last bit, while `((0.1 + 0.2) + 0.3) / 3` reports
/// `0.20000000000000004`.
#[test]
fn forecast_means_do_not_accumulate_rounding_at_an_ordinary_scale() {
    let columns: [&[f64]; 2] = [&[0.1, 0.2, 0.3], &[1.0, 2.0, 3.0]];
    for (name, means) in all_means(&columns) {
        assert_eq!(means, vec![0.2, 2.0], "{name}");
    }
}

// ---------------------------------------------------------------------------
// Defect 2 — the local-level filtering and smoothing variance updates
// ---------------------------------------------------------------------------

/// Filtered variances for three observed steps with the initial, process and
/// observation variances all equal to the first entry, replayed out of crate
/// over the exact rational values of the stored doubles and rounded once to
/// `f64`.
///
/// The recursion is `P = V + Q`, `V' = P R / (P + R)`; with `Q == R == V[-1]`
/// the three filtered variances are `2/3`, `5/8` and `13/21` of that common
/// scale, which is why the entries at one scale sit so close together.
const FILTERED_VARIANCES: [(f64, [f64; 3]); 4] = [
    (3e-162, [2e-162, 1.875e-162, 1.857142857142857e-162]),
    (
        1e-200,
        [6.666666666666667e-201, 6.25e-201, 6.19047619047619e-201],
    ),
    (
        1e200,
        [
            6.666666666666667e199,
            6.249999999999999e199,
            6.190476190476191e199,
        ],
    ),
    (0.4, [0.26666666666666666, 0.25, 0.24761904761904763]),
];

/// `P * R / (P + R)` forms `P * R` first, which leaves the representable range
/// long before the quotient does.
///
/// At `3e-162` the product is subnormal: the first update returned
/// `2.1958473148499844e-162` against a correctly rounded `2e-162`, 9.79% high,
/// and the second `1.8084730969037942e-162` against `1.875e-162`, 3.55% low.
/// At `1e-200` the product underflows to zero and at `1e200` it overflows, in
/// both cases for a filtering problem whose answer is an ordinary number. The
/// positivity guard does not catch the first case, because the wrong answers
/// are positive.
#[test]
fn local_level_filtered_variances_match_a_high_precision_recursion() {
    let observations = [0.5, -0.25, 0.75];
    for (scale, expected_steps) in FILTERED_VARIANCES {
        let filter = rustmc_core::bayesian_forecast::filter_local_level(
            &observations,
            0.0,
            scale,
            scale,
            scale,
        )
        .unwrap_or_else(|error| panic!("filter failed at {scale}: {error}"));
        assert_eq!(
            filter.filtered_variances[0], scale,
            "prior state at {scale}"
        );
        for (step, expected) in expected_steps.into_iter().enumerate() {
            let actual = filter.filtered_variances[step + 1];
            assert!(
                (actual - expected).abs() <= f64::EPSILON * expected.abs(),
                "step {step} at {scale}: {actual} vs {expected}"
            );
        }
    }
    // The two entries the reported symptom names are reproduced exactly, not
    // just to within a rounding of the reference.
    let filter = rustmc_core::bayesian_forecast::filter_local_level(
        &observations,
        0.0,
        3e-162,
        3e-162,
        3e-162,
    )
    .unwrap();
    assert_eq!(filter.filtered_variances[1], 2e-162);
    assert_eq!(filter.filtered_variances[2], 1.875e-162);
}

/// The same defect in the backward-sampling variance, `V Q / (V + Q)`, reached
/// through the public Gibbs sampler.
///
/// Every variance here is `1e-200`, so both products underflow to zero and
/// `validate_positive_variance` turns a perfectly well scaled model into a
/// `NumericalFailure`. Fixing only the filtering update leaves the smoother
/// failing, so this covers both sites.
#[test]
fn local_level_gibbs_runs_where_the_variance_products_underflow() {
    let observations = [1e-100, 2e-100, 1.5e-100, 0.5e-100, 1.2e-100];
    let config = BayesianLocalLevelConfig {
        initial_mean: 0.0,
        initial_variance: 1e-200,
        // mode = scale / (shape + 1) = 1e-200
        process_variance_prior: InverseGammaPrior::new(1.0, 2e-200).unwrap(),
        observation_variance_prior: InverseGammaPrior::new(1.0, 2e-200).unwrap(),
        num_chains: 2,
        num_warmup: 20,
        num_draws: 40,
        thinning: 1,
        seed: 2026,
    };
    let posterior = fit_bayesian_local_level(&observations, &config)
        .unwrap_or_else(|error| panic!("subnormal-scale fit failed: {error}"));
    for draw in posterior.chains.iter().flatten() {
        assert!(draw.process_variance > 0.0 && draw.process_variance.is_finite());
        assert!(draw.observation_variance > 0.0 && draw.observation_variance.is_finite());
        assert!(draw.terminal_level.is_finite());
    }
    let forecast = posterior.forecast(3, 2027).unwrap();
    assert!(forecast
        .observation_means()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}
