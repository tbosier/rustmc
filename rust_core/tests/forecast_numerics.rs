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
///
/// The field under test carries `columns`; every other field of the same result
/// carries a decoy whose means are different, so an accessor that read the
/// wrong field would fail rather than pass on an identical copy.
fn all_means(columns: &[&[f64]]) -> Vec<(&'static str, Vec<f64>)> {
    let paths = chains(columns);
    const DECOY: [f64; 4] = [7.5, -2.5, 0.25, 1.0];
    let decoy_column: &[f64] = &DECOY[..columns[0].len()];
    let decoy = chains(&vec![decoy_column; columns.len()]);
    let local_level = PosteriorPredictiveForecast {
        state_paths: paths.clone(),
        observation_paths: decoy.clone(),
    };
    let local_level_obs = PosteriorPredictiveForecast {
        state_paths: decoy.clone(),
        observation_paths: paths.clone(),
    };
    let trend = TrendPosteriorPredictiveForecast {
        level_paths: paths.clone(),
        slope_paths: decoy.clone(),
        observation_paths: decoy.clone(),
    };
    let trend_slope = TrendPosteriorPredictiveForecast {
        level_paths: decoy.clone(),
        slope_paths: paths.clone(),
        observation_paths: decoy.clone(),
    };
    let trend_obs = TrendPosteriorPredictiveForecast {
        level_paths: decoy.clone(),
        slope_paths: decoy.clone(),
        observation_paths: paths.clone(),
    };
    let seasonal = SeasonalPosteriorPredictiveForecast {
        level_paths: paths.clone(),
        seasonal_paths: decoy.clone(),
        observation_paths: decoy.clone(),
        cumulative_observation_paths: decoy.clone(),
    };
    let seasonal_seasonal = SeasonalPosteriorPredictiveForecast {
        level_paths: decoy.clone(),
        seasonal_paths: paths.clone(),
        observation_paths: decoy.clone(),
        cumulative_observation_paths: decoy.clone(),
    };
    let seasonal_obs = SeasonalPosteriorPredictiveForecast {
        level_paths: decoy.clone(),
        seasonal_paths: decoy.clone(),
        observation_paths: paths.clone(),
        cumulative_observation_paths: decoy.clone(),
    };
    let seasonal_cumulative = SeasonalPosteriorPredictiveForecast {
        level_paths: decoy.clone(),
        seasonal_paths: decoy.clone(),
        observation_paths: decoy.clone(),
        cumulative_observation_paths: paths.clone(),
    };
    let ar = BayesianArForecast {
        conditional_mean_paths: paths.clone(),
        observation_paths: decoy.clone(),
    };
    let ar_obs = BayesianArForecast {
        conditional_mean_paths: decoy,
        observation_paths: paths,
    };
    vec![
        (
            "local_level.state_means",
            local_level.state_means().unwrap(),
        ),
        (
            "local_level.observation_means",
            local_level_obs.observation_means().unwrap(),
        ),
        ("trend.level_means", trend.level_means().unwrap()),
        ("trend.slope_means", trend_slope.slope_means().unwrap()),
        (
            "trend.observation_means",
            trend_obs.observation_means().unwrap(),
        ),
        ("seasonal.level_means", seasonal.level_means().unwrap()),
        (
            "seasonal.seasonal_means",
            seasonal_seasonal.seasonal_means().unwrap(),
        ),
        (
            "seasonal.observation_means",
            seasonal_obs.observation_means().unwrap(),
        ),
        (
            "seasonal.cumulative_observation_means",
            seasonal_cumulative.cumulative_observation_means().unwrap(),
        ),
        (
            "ar.conditional_mean_means",
            ar.conditional_mean_means().unwrap(),
        ),
        ("ar.observation_means", ar_obs.observation_means().unwrap()),
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

/// The same accessors on one ordinary-scale example the running sum got wrong:
/// the exact mean of `[0.1, 0.2, 0.3]` is `0.2` to the last bit, while
/// `((0.1 + 0.2) + 0.3) / 3` reports `0.20000000000000004`.
///
/// This is one example, not a general accuracy guarantee. The centred form
/// rounds once per draw too, and the error in its mean is of order one ulp of
/// the draws' *spread* rather than of the mean — a trade
/// `diagnostics::scaled_moments` documents, and one that reports `0` for
/// `[1, -1, 1e-16, 1e-16]` where a running sum reports the correctly rounded
/// `5e-17`.
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

/// The observation schedule the filtering cases run on. Index one is missing,
/// so the update that skips the observation is covered too.
const FILTER_OBSERVATIONS: [f64; 4] = [0.5, f64::NAN, -0.25, 0.75];

/// `(initial, process, observation)` variances and the four filtered variances
/// they produce on `FILTER_OBSERVATIONS`.
///
/// The recursion is `P = V + Q`, then `V' = P` at the missing step and
/// `V' = P R / (P + R)` elsewhere. Every expectation was produced out of crate,
/// by replaying that recursion over the exact rational values of the stored
/// doubles with Python's `fractions` and rounding each step once to `f64`
/// through `decimal` at 400 significant digits.
///
/// The last three rows are what make the case discriminating: the variances
/// differ from one another, so an update that multiplied by the process
/// variance where it should multiply by the observation variance would not
/// survive, and the `1e-200` state against `1e120` noise is the regime where
/// dividing before multiplying is the ordering that fails.
#[allow(clippy::type_complexity)]
const FILTERED_VARIANCES: [((f64, f64, f64), [f64; 4]); 7] = [
    (
        (3e-162, 3e-162, 3e-162),
        [2e-162, 5e-162, 2.181818181818182e-162, 1.9e-162],
    ),
    (
        (1e-200, 1e-200, 1e-200),
        [
            6.666666666666667e-201,
            1.6666666666666665e-200,
            7.272727272727273e-201,
            6.333333333333333e-201,
        ],
    ),
    (
        (1e200, 1e200, 1e200),
        [
            6.666666666666667e199,
            1.6666666666666667e200,
            7.272727272727273e199,
            6.333333333333333e199,
        ],
    ),
    (
        (0.4, 0.4, 0.4),
        [
            0.26666666666666666,
            0.6666666666666667,
            0.29090909090909095,
            0.25333333333333335,
        ],
    ),
    // State variance far below the noise variance: the filtered variance is
    // essentially the predicted one, and `P / (P + R)` is the quotient that
    // cannot be formed first.
    ((1e-200, 1e-200, 1e120), [2e-200, 3e-200, 4e-200, 5e-200]),
    // And the other way round.
    ((1e120, 1e120, 1e-200), [1e-200, 1e120, 1e-200, 1e-200]),
    (
        (0.25, 1.5, 0.0625),
        [
            0.0603448275862069,
            1.5603448275862069,
            0.061249137336093856,
            0.06009430203214238,
        ],
    ),
];

/// `P R / (P + R)` has three orderings and two of them leave the representable
/// range on inputs whose answer is an ordinary number.
///
/// Forming `P R` first: at `P = 6e-162, R = 3e-162` the product is subnormal
/// and the first update returned `2.1958473148499844e-162` against a correctly
/// rounded `2e-162`, 9.79% high, and the second `1.8084730969037942e-162`
/// against `1.875e-162`, 3.55% low; at `1e-200` it underflows to zero and at
/// `1e200` it overflows. Forming `P / (P + R)` first instead fails in the
/// opposite corner: at `P = 1e-200, R = 1e120` the quotient is subnormal and
/// the answer loses eleven significant digits, and at `R = 1e200` it is zero.
/// The positivity guard catches none of the inexact cases, because the wrong
/// answers are positive.
#[test]
fn local_level_filtered_variances_match_a_high_precision_recursion() {
    for ((initial, process, observation), expected_steps) in FILTERED_VARIANCES {
        let filter = rustmc_core::bayesian_forecast::filter_local_level(
            &FILTER_OBSERVATIONS,
            0.0,
            initial,
            process,
            observation,
        )
        .unwrap_or_else(|error| {
            panic!("filter failed at ({initial}, {process}, {observation}): {error}")
        });
        assert_eq!(filter.filtered_variances[0], initial);
        assert_eq!(
            filter.filtered_variances.len(),
            FILTER_OBSERVATIONS.len() + 1
        );
        for (step, expected) in expected_steps.into_iter().enumerate() {
            let actual = filter.filtered_variances[step + 1];
            assert!(
                (actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs(),
                "step {step} at ({initial}, {process}, {observation}): {actual} vs {expected}"
            );
        }
    }
    // The two entries the reported symptom names are reproduced exactly, not
    // only to within a rounding of the reference.
    let filter = rustmc_core::bayesian_forecast::filter_local_level(
        &FILTER_OBSERVATIONS,
        0.0,
        3e-162,
        3e-162,
        3e-162,
    )
    .unwrap();
    assert_eq!(filter.filtered_variances[1], 2e-162);
    // The missing step at index one only propagates, so it is exact too.
    assert_eq!(filter.filtered_variances[2], 5e-162);
}

/// The filtered levels, which the variance recursion drives through the gain.
///
/// With all three variances equal to one the filtered variances are
/// `1, 2/3, 5/3, 8/11, 19/30` and the gains `P / (P + R)` are `2/3`, nothing
/// across the missing step, `8/11` and `19/30`, so the levels are rational
/// combinations of the observations that can be written down in closed form.
#[test]
fn local_level_filtered_levels_follow_the_closed_form_gains() {
    let filter = rustmc_core::bayesian_forecast::filter_local_level(
        &FILTER_OBSERVATIONS,
        0.0,
        1.0,
        1.0,
        1.0,
    )
    .unwrap();
    let m1 = 0.0 + (2.0 / 3.0) * (0.5 - 0.0);
    let m2 = m1;
    let m3 = m2 + (8.0 / 11.0) * (-0.25 - m2);
    let m4 = m3 + (19.0 / 30.0) * (0.75 - m3);
    let expected = [0.0, m1, m2, m3, m4];
    for (step, want) in expected.into_iter().enumerate() {
        let got = filter.filtered_means[step];
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(1.0),
            "level {step}: {got} vs {want}"
        );
    }
    for (step, want) in [1.0, 2.0 / 3.0, 5.0 / 3.0, 8.0 / 11.0, 19.0 / 30.0]
        .into_iter()
        .enumerate()
    {
        let got = filter.filtered_variances[step];
        assert!(
            (got - want).abs() <= 2.0 * f64::EPSILON * want,
            "variance {step}: {got} vs {want}"
        );
    }
}

/// The same defect in the backward-sampling variance, `V Q / (V + Q)`, reached
/// through the public Gibbs sampler.
///
/// The initial variance and both prior modes are `1e-200`, so on the first
/// iteration every product underflows to zero and `validate_positive_variance`
/// turns a well scaled model into a `NumericalFailure`. Later iterations draw
/// their variances, so they are not all `1e-200`, but they stay in that region.
/// Reverting either of the two variance sites brings the failure back, so this
/// covers the smoother as well as the filter.
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
        // The data are all of order 1e-100, so a level that had lost the scale
        // would show up here even though it stayed finite.
        assert!(draw.terminal_level.abs() < 1e-95, "{}", draw.terminal_level);
    }
    let forecast = posterior.forecast(3, 2027).unwrap();
    assert!(forecast
        .observation_means()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}

/// The new public filter validates what it documents.
#[test]
fn filter_local_level_rejects_an_unusable_starting_point() {
    use rustmc_core::bayesian_forecast::filter_local_level;
    assert!(filter_local_level(&FILTER_OBSERVATIONS, f64::NAN, 1.0, 1.0, 1.0).is_err());
    assert!(filter_local_level(&[], f64::NAN, 1.0, 1.0, 1.0).is_err());
    assert!(filter_local_level(&FILTER_OBSERVATIONS, 0.0, 0.0, 1.0, 1.0).is_err());
    assert!(filter_local_level(&FILTER_OBSERVATIONS, 0.0, 1.0, -1.0, 1.0).is_err());
    assert!(filter_local_level(&FILTER_OBSERVATIONS, 0.0, 1.0, 1.0, f64::NAN).is_err());
    // An empty series is a filter with nothing but its starting point.
    let empty = filter_local_level(&[], 2.5, 1.0, 1.0, 1.0).unwrap();
    assert_eq!(empty.filtered_means, vec![2.5]);
    assert_eq!(empty.filtered_variances, vec![1.0]);
}
