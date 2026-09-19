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
use rustmc_core::bayesian_forecast::PosteriorPredictiveForecast;
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
