//! Bayesian inference and posterior prediction for a local-linear-trend model.
//!
//! The model is
//!
//! ```text
//! [level[-1], slope[-1]] ~ Normal(initial_mean, initial_covariance)
//! level[t] = level[t - 1] + slope[t - 1] + Normal(0, level_variance)
//! slope[t] = slope[t - 1] + Normal(0, slope_variance)
//! y[t] = level[t] + Normal(0, observation_variance)
//! ```
//!
//! All three variances have caller-specified inverse-gamma priors. The fitted
//! posterior is sampled by a seeded Gibbs sampler over the three variances with
//! the states integrated out: each variance in turn takes one slice-sampling
//! step on its conditional given the other two, whose likelihood is computed
//! exactly by a two-state Kalman filter. Retained draws add the terminal state,
//! drawn from its filtered distribution. `NaN` observations retain their
//! scheduled time positions while infinities are rejected.
//!
//! An earlier version drew the whole state path by forward filtering, backward
//! sampling (FFBS) and then each variance from its conjugate inverse-gamma
//! conditional given the states. That mixes slowly whenever the states pin
//! their variances down, and badly when a variance is near zero; see
//! `benchmarks/llt_variance_mixing.py`.
//!
//! Independent chains execute on the active Rayon pool (or its global pool),
//! while indexed collection preserves deterministic chain ordering.

use crate::diagnostics::{fit_warnings, DiagnosticsReport, FitWarning, SamplerTelemetry};
use crate::forecast_common::{
    check_forecast_size, invalid_config, numerical, overdispersed_positive, path_means,
    path_quantiles, require_finite_observations, run_gibbs_chains, simulate_draws,
    slice_sample_positive, split_paths, standard_normal, validate_positive_variance,
    BayesianForecastError, ForecastQuantile, GibbsSchedule, InverseGammaPrior,
    TREND_FIT_SEED_DOMAIN as FIT_SEED_DOMAIN, TREND_FORECAST_SEED_DOMAIN as FORECAST_SEED_DOMAIN,
};
use crate::forecast_diagnostics::parameter_diagnostics;
use rand_chacha::ChaCha8Rng;

const SYMMETRY_TOLERANCE: f64 = 1e-10;

/// How fit warnings name this model's sampler; `sampler_stats["sampler"]`
/// in the Python bindings uses the same words.
const SAMPLER_NAME: &str = "Gibbs over variances with states integrated out (slice steps)";

/// Configuration for a fitted Bayesian local-linear-trend model.
#[derive(Debug, Clone)]
pub struct BayesianLocalLinearTrendConfig {
    /// Mean of `[level[-1], slope[-1]]`.
    pub initial_mean: [f64; 2],
    /// Row-major covariance of `[level[-1], slope[-1]]`.
    pub initial_covariance: [f64; 4],
    pub level_variance_prior: InverseGammaPrior,
    pub slope_variance_prior: InverseGammaPrior,
    pub observation_variance_prior: InverseGammaPrior,
    pub num_chains: usize,
    pub num_warmup: usize,
    pub num_draws: usize,
    pub thinning: usize,
    pub seed: u64,
}

impl BayesianLocalLinearTrendConfig {
    fn validate(&self) -> Result<GibbsSchedule, BayesianForecastError> {
        if self.initial_mean.iter().any(|value| !value.is_finite()) {
            return Err(invalid_config(
                "initial mean must contain only finite values",
            ));
        }
        validate_covariance(self.initial_covariance)?;
        self.level_variance_prior.validate("level-variance prior")?;
        self.slope_variance_prior.validate("slope-variance prior")?;
        self.observation_variance_prior
            .validate("observation-variance prior")?;
        GibbsSchedule::new(
            self.num_chains,
            self.num_warmup,
            self.num_draws,
            self.thinning,
        )
    }
}

/// One joint posterior draw for the local-linear-trend model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLinearTrendPosteriorDraw {
    pub level_variance: f64,
    pub slope_variance: f64,
    pub observation_variance: f64,
    pub terminal_level: f64,
    pub terminal_slope: f64,
}

/// Joint posterior draws indexed as `[chain][draw]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalLinearTrendPosterior {
    pub chains: Vec<Vec<LocalLinearTrendPosteriorDraw>>,
}

impl LocalLinearTrendPosterior {
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let names = [
            "level_variance",
            "slope_variance",
            "observation_variance",
            "terminal_level",
            "terminal_slope",
        ]
        .map(String::from);
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        vec![
                            d.level_variance,
                            d.slope_variance,
                            d.observation_variance,
                            d.terminal_level,
                            d.terminal_slope,
                        ]
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }

    /// Every failed or unavailable check of this fit (see [`fit_warnings`]).
    /// With the states integrated out the sampler mixes its variances well
    /// in every setting of `benchmarks/state_variance_mixing.py`, so it
    /// carries no `known_limitation` record.
    pub fn warnings(&self) -> Vec<FitWarning> {
        fit_warnings(
            &self.diagnostics(),
            SamplerTelemetry::NonHamiltonian {
                sampler: SAMPLER_NAME,
                constant: &[],
            },
        )
    }

    /// Pair every posterior draw with one coherent future state/observation path.
    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<TrendPosteriorPredictiveForecast, BayesianForecastError> {
        if horizon == 0 {
            return Err(invalid_config("forecast horizon must be positive"));
        }
        if self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(invalid_config(
                "every posterior chain must contain at least one draw",
            ));
        }

        check_forecast_size("local-linear-trend forecast", &self.chains, horizon, 3)?;

        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &LocalLinearTrendPosteriorDraw, rng| {
                validate_positive_variance("level", draw.level_variance)?;
                validate_positive_variance("slope", draw.slope_variance)?;
                validate_positive_variance("observation", draw.observation_variance)?;
                if !draw.terminal_level.is_finite() || !draw.terminal_slope.is_finite() {
                    return Err(numerical("posterior terminal state is not finite"));
                }

                let level_sd = draw.level_variance.sqrt();
                let slope_sd = draw.slope_variance.sqrt();
                let observation_sd = draw.observation_variance.sqrt();
                let mut level = draw.terminal_level;
                let mut slope = draw.terminal_slope;
                let mut levels = Vec::with_capacity(horizon);
                let mut slopes = Vec::with_capacity(horizon);
                let mut observations = Vec::with_capacity(horizon);
                for _ in 0..horizon {
                    // Both innovations are applied to F x[t - 1], independently.
                    level += slope + standard_normal(rng) * level_sd;
                    slope += standard_normal(rng) * slope_sd;
                    let observation = level + standard_normal(rng) * observation_sd;
                    if !level.is_finite() || !slope.is_finite() || !observation.is_finite() {
                        return Err(numerical("posterior predictive simulation overflowed"));
                    }
                    levels.push(level);
                    slopes.push(slope);
                    observations.push(observation);
                }
                Ok([levels, slopes, observations])
            },
        )?;
        let [level_paths, slope_paths, observation_paths] = split_paths(per_draw);

        Ok(TrendPosteriorPredictiveForecast {
            level_paths,
            slope_paths,
            observation_paths,
        })
    }
}

/// Coherent posterior-predictive paths indexed as `[chain][draw][step]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrendPosteriorPredictiveForecast {
    pub level_paths: Vec<Vec<Vec<f64>>>,
    pub slope_paths: Vec<Vec<Vec<f64>>>,
    pub observation_paths: Vec<Vec<Vec<f64>>>,
}

impl TrendPosteriorPredictiveForecast {
    pub fn horizon(&self) -> usize {
        self.observation_paths
            .first()
            .and_then(|chain| chain.first())
            .map_or(0, Vec::len)
    }

    pub fn level_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.level_paths)
    }

    pub fn slope_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.slope_paths)
    }

    pub fn observation_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.observation_paths)
    }

    pub fn level_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.level_paths, probabilities)
    }

    pub fn slope_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.slope_paths, probabilities)
    }

    pub fn observation_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.observation_paths, probabilities)
    }
}

/// Fit a Bayesian local-linear-trend model by Gibbs sampling over its three
/// variances with the states integrated out.
///
/// Each sweep updates the level, slope and observation variances in turn,
/// each by one slice-sampling step on its conditional given the other two,
/// `prior(q) * p(y | variances)`, where `p(y | variances)` is the Kalman
/// filter likelihood with every state integrated out. This targets the same
/// posterior as drawing the states and then each variance given them, but no
/// update conditions on the states, so the variances do not inherit the
/// states' autocorrelation. A retained draw adds the terminal state from
/// `p(x[T - 1] | variances, y)`, the filter's last filtered distribution.
pub fn fit_bayesian_local_linear_trend(
    observations: &[f64],
    config: &BayesianLocalLinearTrendConfig,
) -> Result<LocalLinearTrendPosterior, BayesianForecastError> {
    let schedule = config.validate()?;
    require_finite_observations(observations, 3, "local-linear-trend")?;
    // Five values per draw. The filter keeps only its current moments, so a
    // chain's working state does not grow with the series.
    schedule.check_fit_size("local-linear-trend fit", &[5], &[6])?;
    let priors = [
        config.level_variance_prior,
        config.slope_variance_prior,
        config.observation_variance_prior,
    ];
    let chains = run_gibbs_chains(
        &schedule,
        config.seed,
        FIT_SEED_DOMAIN,
        |rng| {
            Ok::<_, BayesianForecastError>(
                priors.map(|prior| overdispersed_positive(prior.mode(), rng)),
            )
        },
        |variances, rng, retain| {
            for (index, prior) in priors.iter().enumerate() {
                let others = *variances;
                variances[index] = slice_sample_positive(
                    others[index],
                    |candidate| {
                        let mut trial = others;
                        trial[index] = candidate;
                        prior.log_density(candidate)
                            + log_marginal_likelihood(
                                observations,
                                config.initial_mean,
                                config.initial_covariance,
                                trial,
                            )
                    },
                    rng,
                )?;
            }
            if !retain {
                return Ok(None);
            }
            let [level_variance, slope_variance, observation_variance] = *variances;
            let [terminal_level, terminal_slope] = sample_terminal_state(
                observations,
                config.initial_mean,
                config.initial_covariance,
                *variances,
                rng,
            )?;
            Ok(Some(LocalLinearTrendPosteriorDraw {
                level_variance,
                slope_variance,
                observation_variance,
                terminal_level,
                terminal_slope,
            }))
        },
    )?;
    Ok(LocalLinearTrendPosterior { chains })
}

/// Log density of the observations given the level, slope and observation
/// variances, with every state integrated out, by the prediction-error
/// decomposition of the Kalman filter. Returns `-inf` where the filter breaks
/// down, which the slice sampler treats as zero density.
fn log_marginal_likelihood(
    observations: &[f64],
    initial_mean: [f64; 2],
    initial_covariance: [f64; 4],
    [level_variance, slope_variance, observation_variance]: [f64; 3],
) -> f64 {
    let Ok(mut covariance) = StateCovariance::new(symmetrized(initial_covariance)) else {
        return f64::NEG_INFINITY;
    };
    let mut mean = initial_mean;
    let mut log_likelihood = 0.0;
    for &observation in observations {
        let step = kalman_step(
            mean,
            covariance,
            observation,
            level_variance,
            slope_variance,
            observation_variance,
        );
        if let Some((innovation, innovation_variance)) = step.innovation {
            if !innovation_variance.is_finite() || innovation_variance <= 0.0 {
                return f64::NEG_INFINITY;
            }
            // Standardise before squaring: the square of a large innovation
            // can overflow where its ratio to the variance does not.
            let standardized = innovation / innovation_variance.sqrt();
            log_likelihood -=
                0.5 * (LOG_2_PI + innovation_variance.ln() + standardized * standardized);
        }
        mean = step.filtered_mean;
        covariance = step.filtered_covariance;
    }
    if log_likelihood.is_finite() {
        log_likelihood
    } else {
        f64::NEG_INFINITY
    }
}

const LOG_2_PI: f64 = 1.837_877_066_409_345_5;

/// A state covariance as `L D L'` with `L = [[1, 0], [beta, 1]]` and
/// `D = diag(level, slope_given_level)`: the level's variance, the slope's
/// regression on the level, and the slope's variance given the level.
///
/// Forming the covariance itself lost the slope variance to cancellation once
/// the initial covariance was many orders of magnitude above the variances:
/// each predicted entry was a huge number plus a small one, and the update
/// keeps only the small part. In this form an observation of the level leaves
/// `beta` and `slope_given_level` unchanged and scales `level`, and the
/// predicted `slope_given_level` and `level` are sums of non-negative terms.
/// What can still cancel is `1 + beta`, and the predicted level/slope
/// covariance when `beta` lies between -1 and 0.
#[derive(Debug, Clone, Copy)]
struct StateCovariance {
    level: f64,
    beta: f64,
    slope_given_level: f64,
}

impl StateCovariance {
    /// Fails unless the level variance and the slope's variance given the
    /// level are positive and every part is finite, which for a symmetric
    /// 2 x 2 matrix is positive definiteness. The slope's variance given the
    /// level is the determinant over the level variance. The determinant
    /// comes from Kahan's fused multiply-add algorithm, on the matrix scaled
    /// by a power of two so that its largest entry is near one, which keeps it
    /// accurate to a few rounding errors relative to itself however much
    /// `level slope` and `cross^2` cancel. `slope - cross * (cross / level)`,
    /// used before, was 60% off for `[[0.1, 0.3], [0.3, 0.9]]` and zero for a
    /// positive definite `[[0.3, -0.7], [-0.7, 1.6333333333333333]]`.
    fn new(covariance: [f64; 4]) -> Result<Self, ()> {
        let [level, cross, _, slope] = covariance;
        let largest = level.abs().max(slope.abs()).max(cross.abs());
        let slope_given_level = if largest.is_finite() && largest > 0.0 {
            // Multiplying by a power of two is exact, in two halves so that
            // neither factor overflows.
            let exponent = -(largest.log2().floor() as i32);
            let half = 2f64.powi(exponent / 2);
            let rest = 2f64.powi(exponent - exponent / 2);
            let [scaled_level, scaled_cross, scaled_slope] =
                [level, cross, slope].map(|value| value * half * rest);
            let square = scaled_cross * scaled_cross;
            let product = scaled_level * scaled_slope;
            if product.is_finite() && product.abs() >= f64::MIN_POSITIVE / f64::EPSILON {
                let rounding = (-scaled_cross).mul_add(scaled_cross, square);
                (scaled_level.mul_add(scaled_slope, -square) + rounding)
                    / scaled_level
                    / half
                    / rest
            } else {
                // `level slope` is far below `largest^2`, so for a positive
                // definite matrix `cross^2` is too, and the plain form does
                // not cancel; unscaled, its small entries keep their digits.
                slope - cross * (cross / level)
            }
        } else {
            f64::NAN
        };
        let candidate = Self {
            level,
            beta: cross / level,
            slope_given_level,
        };
        if candidate.is_valid() {
            Ok(candidate)
        } else {
            Err(())
        }
    }

    /// The covariance of `[level + slope + eta, slope + zeta]`, with `eta`
    /// and `zeta` independent of variances `level_variance` and
    /// `slope_variance`. Writing `slope = beta level + e`, the predicted
    /// level's noise-free part is `x = (1 + beta) level + e` and the slope's
    /// is `w = beta level + e`; `Var(w | x)` is `level slope_given_level /
    /// Var(x)` exactly, which avoids subtracting `Cov(w, x)^2 / Var(x)`.
    fn predicted(self, level_variance: f64, slope_variance: f64) -> Self {
        let Self {
            level,
            beta,
            slope_given_level,
        } = self;
        let rise = 1.0 + beta;
        let x_variance = scaled_product(&[rise, rise, level], &[]) + slope_given_level;
        let cross = scaled_product(&[beta, rise, level], &[]) + slope_given_level;
        let predicted_level = x_variance + level_variance;
        Self {
            level: predicted_level,
            beta: cross / predicted_level,
            slope_given_level: slope_variance
                + scaled_product(&[level, slope_given_level], &[x_variance])
                + scaled_product(
                    &[cross, cross, level_variance],
                    &[x_variance, predicted_level],
                ),
        }
    }

    fn is_valid(&self) -> bool {
        self.level.is_finite()
            && self.beta.is_finite()
            && self.slope_given_level.is_finite()
            && self.level > 0.0
            && self.slope_given_level > 0.0
    }

    #[cfg(test)]
    fn matrix(&self) -> [f64; 4] {
        let cross = self.beta * self.level;
        [
            self.level,
            cross,
            cross,
            self.beta * cross + self.slope_given_level,
        ]
    }
}

/// The product of `numerators` divided by the product of `denominators`.
///
/// Multiplying and dividing in turn is exact to rounding and is what this
/// returns whenever every intermediate value stays in the normal range. When
/// one would overflow or underflow although the result need not, as with a
/// variance of 1e-100 against one of 1e300, the magnitudes are combined as
/// logarithms instead, at a relative cost of about 1e-13.
fn scaled_product(numerators: &[f64], denominators: &[f64]) -> f64 {
    if numerators
        .iter()
        .chain(denominators)
        .any(|x| *x == 0.0 || !x.is_finite())
    {
        // Zeros, infinities and NaNs give what the plain product gives,
        // except that a zero numerator over finite, nonzero denominators is
        // zero even where their product would underflow to zero.
        if numerators.contains(&0.0)
            && numerators.iter().all(|x| x.is_finite())
            && denominators.iter().all(|x| *x != 0.0 && x.is_finite())
        {
            return 0.0;
        }
        return numerators.iter().product::<f64>() / denominators.iter().product::<f64>();
    }
    let in_range = |value: f64| value.is_finite() && value.abs() >= f64::MIN_POSITIVE;
    let mut value = 1.0;
    let mut direct = true;
    for index in 0..numerators.len().max(denominators.len()) {
        if let Some(&numerator) = numerators.get(index) {
            value *= numerator;
            direct &= in_range(value);
        }
        if let Some(&denominator) = denominators.get(index) {
            value /= denominator;
            direct &= in_range(value);
        }
        if !direct {
            break;
        }
    }
    if direct {
        return value;
    }
    let sign: f64 = numerators
        .iter()
        .chain(denominators)
        .map(|value| value.signum())
        .product();
    let log_magnitude = numerators.iter().map(|value| value.abs().ln()).sum::<f64>()
        - denominators
            .iter()
            .map(|value| value.abs().ln())
            .sum::<f64>();
    sign * log_magnitude.exp()
}

/// One Kalman filter prediction and, for a non-missing observation, update.
struct KalmanStep {
    predicted_mean: [f64; 2],
    predicted_covariance: StateCovariance,
    filtered_mean: [f64; 2],
    filtered_covariance: StateCovariance,
    /// The one-step prediction error and its variance; `None` when missing.
    innovation: Option<(f64, f64)>,
}

fn kalman_step(
    mean: [f64; 2],
    covariance: StateCovariance,
    observation: f64,
    level_variance: f64,
    slope_variance: f64,
    observation_variance: f64,
) -> KalmanStep {
    let predicted_mean = [mean[0] + mean[1], mean[1]];
    let predicted_covariance = covariance.predicted(level_variance, slope_variance);
    if observation.is_nan() {
        return KalmanStep {
            predicted_mean,
            predicted_covariance,
            filtered_mean: predicted_mean,
            filtered_covariance: predicted_covariance,
            innovation: None,
        };
    }
    let innovation_variance = predicted_covariance.level + observation_variance;
    let innovation = observation - predicted_mean[0];
    // The gain P H' / S is [level, beta level] / S.
    let level_gain = predicted_covariance.level / innovation_variance;
    KalmanStep {
        predicted_mean,
        predicted_covariance,
        filtered_mean: [
            predicted_mean[0] + level_gain * innovation,
            predicted_mean[1] + predicted_covariance.beta * level_gain * innovation,
        ],
        filtered_covariance: StateCovariance {
            level: scaled_product(
                &[predicted_covariance.level, observation_variance],
                &[innovation_variance],
            ),
            ..predicted_covariance
        },
        innovation: Some((innovation, innovation_variance)),
    }
}

/// Draw the terminal state `x[T - 1]` from `p(x[T - 1] | variances, y)`: run
/// the Kalman filter to the end, checking every predicted and filtered moment,
/// and sample the last filtered distribution.
fn sample_terminal_state(
    observations: &[f64],
    initial_mean: [f64; 2],
    initial_covariance: [f64; 4],
    variances: [f64; 3],
    rng: &mut ChaCha8Rng,
) -> Result<[f64; 2], BayesianForecastError> {
    let [level_variance, slope_variance, observation_variance] = variances;
    validate_positive_variance("level", level_variance)?;
    validate_positive_variance("slope", slope_variance)?;
    validate_positive_variance("observation", observation_variance)?;

    let mut mean = initial_mean;
    let mut covariance = StateCovariance::new(symmetrized(initial_covariance))
        .map_err(|_| invalid_config(NOT_POSITIVE_DEFINITE))?;
    for (time, &observation) in observations.iter().enumerate() {
        let step = kalman_step(
            mean,
            covariance,
            observation,
            level_variance,
            slope_variance,
            observation_variance,
        );
        validate_computed_state(
            "predicted",
            time,
            step.predicted_mean,
            step.predicted_covariance,
        )?;
        if let Some((_, innovation_variance)) = step.innovation {
            validate_positive_variance("innovation", innovation_variance)?;
        }
        validate_computed_state(
            "filtered",
            time,
            step.filtered_mean,
            step.filtered_covariance,
        )?;
        mean = step.filtered_mean;
        covariance = step.filtered_covariance;
    }
    let level_deviation = covariance.level.sqrt();
    let z0 = standard_normal(rng);
    let z1 = standard_normal(rng);
    let draw = [
        mean[0] + level_deviation * z0,
        mean[1] + covariance.beta * level_deviation * z0 + covariance.slope_given_level.sqrt() * z1,
    ];
    if draw.iter().any(|value| !value.is_finite()) {
        return Err(numerical("normal simulation produced a non-finite draw"));
    }
    Ok(draw)
}

const NOT_POSITIVE_DEFINITE: &str = "initial covariance must be strictly positive definite";

fn validate_covariance(covariance: [f64; 4]) -> Result<(), BayesianForecastError> {
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(invalid_config(
            "initial covariance must contain only finite values",
        ));
    }
    // Measure asymmetry relative to these state variances, without an
    // absolute floor that changes validation when the state units change.
    let scale = (covariance[0].abs().sqrt() * covariance[3].abs().sqrt())
        .max(covariance[1].abs())
        .max(covariance[2].abs());
    if (covariance[1] - covariance[2]).abs() > SYMMETRY_TOLERANCE * scale {
        return Err(invalid_config("initial covariance must be symmetric"));
    }
    StateCovariance::new(symmetrized(covariance))
        .map_err(|_| invalid_config(NOT_POSITIVE_DEFINITE))?;
    Ok(())
}

fn validate_computed_state(
    name: &str,
    time: usize,
    mean: [f64; 2],
    covariance: StateCovariance,
) -> Result<(), BayesianForecastError> {
    if mean.iter().any(|value| !value.is_finite()) || !covariance.is_valid() {
        return Err(numerical(format!(
            "{name} state at time {time} is not finite or its covariance is not \
             positive definite"
        )));
    }
    Ok(())
}

fn symmetrized(matrix: [f64; 4]) -> [f64; 4] {
    let off_diagonal = 0.5 * (matrix[1] + matrix[2]);
    [matrix[0], off_diagonal, off_diagonal, matrix[3]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn initial_covariance_symmetry_is_independent_of_state_units() {
        for scale in [1e-20, 1e-14, 1.0, 1e20] {
            let mut config = config(12);
            config.initial_covariance = [scale, 0.0, 0.5 * scale, scale];
            assert!(matches!(
                config.validate(),
                Err(BayesianForecastError::InvalidConfiguration(_))
            ));
            config.initial_covariance = [scale, 0.5 * scale, 0.5 * scale, scale];
            config.validate().unwrap();
        }
        let mut config = config(12);
        config.initial_covariance = [1e-14, 5e-8, 5e-8, 1.0];
        config.validate().unwrap();
    }

    fn config(seed: u64) -> BayesianLocalLinearTrendConfig {
        BayesianLocalLinearTrendConfig {
            initial_mean: [0.0, 0.0],
            initial_covariance: [4.0, 0.2, 0.2, 1.0],
            level_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 0.25,
            },
            slope_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 0.05,
            },
            observation_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 0.5,
            },
            num_chains: 2,
            num_warmup: 100,
            num_draws: 150,
            thinning: 1,
            seed,
        }
    }

    #[test]
    fn fit_and_forecast_are_bitwise_identical_across_rayon_pool_sizes() {
        let observations = [0.1, 0.3, f64::NAN, 1.0, 1.1, 1.7, 1.9];
        let config = config(10);
        let run = || {
            let posterior = fit_bayesian_local_linear_trend(&observations, &config).unwrap();
            let forecast = posterior.forecast(5, 23).unwrap();
            (posterior, forecast)
        };
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        let multi_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(run);
        assert_eq!(single_thread, multi_thread);
    }

    #[test]
    fn posterior_and_forecast_are_seeded_shaped_and_ordered() {
        let observations = [0.1, 0.3, f64::NAN, 1.0, 1.1, 1.7, 1.9];
        let fitted = fit_bayesian_local_linear_trend(&observations, &config(9)).unwrap();
        assert_eq!(
            fitted,
            fit_bayesian_local_linear_trend(&observations, &config(9)).unwrap()
        );
        assert_eq!(fitted.chains.len(), 2);
        assert!(fitted.chains.iter().all(|chain| chain.len() == 150));
        assert_ne!(fitted.chains[0], fitted.chains[1]);

        let forecast = fitted.forecast(5, 22).unwrap();
        assert_eq!(forecast, fitted.forecast(5, 22).unwrap());
        assert_eq!(forecast.horizon(), 5);
        for paths in [
            &forecast.level_paths,
            &forecast.slope_paths,
            &forecast.observation_paths,
        ] {
            assert_eq!(paths.len(), 2);
            assert!(paths.iter().all(|chain| chain.len() == 150));
            assert!(paths.iter().flatten().all(|path| path.len() == 5));
        }
        for quantiles in [
            forecast.level_quantiles(&[0.05, 0.5, 0.95]).unwrap(),
            forecast.slope_quantiles(&[0.05, 0.5, 0.95]).unwrap(),
            forecast.observation_quantiles(&[0.05, 0.5, 0.95]).unwrap(),
        ] {
            for step in 0..5 {
                assert!(quantiles[0].values[step] <= quantiles[1].values[step]);
                assert!(quantiles[1].values[step] <= quantiles[2].values[step]);
            }
        }
    }

    #[test]
    fn validation_rejects_bad_inputs_and_covariances() {
        let valid = config(1);
        assert!(fit_bayesian_local_linear_trend(&[], &valid).is_err());
        assert!(fit_bayesian_local_linear_trend(&[1.0, 2.0], &valid).is_err());
        assert!(fit_bayesian_local_linear_trend(&[1.0, 2.0, f64::INFINITY], &valid).is_err());

        let mut asymmetric = valid.clone();
        asymmetric.initial_covariance = [1.0, 0.2, 0.1, 1.0];
        assert!(fit_bayesian_local_linear_trend(&[1.0, 2.0, 3.0], &asymmetric).is_err());
        let mut singular = valid.clone();
        singular.initial_covariance = [1.0, 1.0, 1.0, 1.0];
        assert!(fit_bayesian_local_linear_trend(&[1.0, 2.0, 3.0], &singular).is_err());
        let mut invalid_prior = valid;
        invalid_prior.slope_variance_prior.scale = 0.0;
        assert!(fit_bayesian_local_linear_trend(&[1.0, 2.0, 3.0], &invalid_prior).is_err());
    }

    /// The general state-space model with this module's trend dynamics.
    fn general_trend_model(
        initial_mean: [f64; 2],
        initial_covariance: [f64; 4],
        [level_variance, slope_variance, observation_variance]: [f64; 3],
    ) -> crate::state_space::LinearGaussianStateSpace {
        crate::state_space::LinearGaussianStateSpace::new(
            2,
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![level_variance, 0.0, 0.0, slope_variance],
            observation_variance,
            initial_mean.to_vec(),
            initial_covariance.to_vec(),
        )
        .unwrap()
    }

    #[test]
    fn marginal_likelihood_matches_the_general_kalman_filter() {
        let observations = [0.2, 0.5, f64::NAN, 0.9, 1.6, f64::NAN, f64::NAN, 2.4];
        let initial_mean = [-0.1, 0.15];
        let initial_covariance = [1.4, 0.25, 0.25, 0.8];
        for variances in [
            [0.3, 0.08, 0.5],
            [1e-6, 0.02, 0.3],
            [0.05, 1e-7, 0.3],
            [0.2, 0.01, 1e-5],
            [40.0, 3.0, 25.0],
        ] {
            let expected = general_trend_model(initial_mean, initial_covariance, variances)
                .filter(&observations)
                .unwrap()
                .log_likelihood;
            let actual =
                log_marginal_likelihood(&observations, initial_mean, initial_covariance, variances);
            assert!(
                (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                "{variances:?}: {actual} vs {expected}"
            );
        }
        // A far-off initial level with a diffuse prior squares to more than
        // f64::MAX, but standardised it does not.
        let offset = [0.0, 1.0, 2.0, 3.0, 4.0];
        let far = [1e155, 0.0];
        let wide = [1e300, 0.0, 0.0, 1.0];
        let expected = general_trend_model(far, wide, [0.25; 3])
            .filter(&offset)
            .unwrap()
            .log_likelihood;
        let actual = log_marginal_likelihood(&offset, far, wide, [0.25; 3]);
        assert!(expected.is_finite());
        assert!(
            (actual - expected).abs() <= 1e-10 * expected.abs(),
            "{actual} vs {expected}"
        );

        // Where the filter cannot be evaluated the slice sampler must see zero
        // density rather than a NaN.
        assert_eq!(
            log_marginal_likelihood(
                &observations,
                initial_mean,
                initial_covariance,
                [f64::INFINITY, 0.1, 0.1],
            ),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn terminal_state_moments_match_existing_kalman_smoother() {
        let observations = [0.2, 0.5, 0.9, f64::NAN, f64::NAN];
        let initial_mean = [-0.1, 0.15];
        let initial_covariance = [1.4, 0.25, 0.25, 0.8];
        let variances = [0.3, 0.08, 0.5];
        let smoother = general_trend_model(initial_mean, initial_covariance, variances)
            .smooth(&observations)
            .unwrap();
        let expected_mean = smoother.smoothed_means.last().unwrap();
        let expected_covariance = smoother.smoothed_covariances.last().unwrap();

        let draws = 30_000;
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut sum = [0.0; 2];
        let mut products = [0.0; 4];
        for _ in 0..draws {
            let state = sample_terminal_state(
                &observations,
                initial_mean,
                initial_covariance,
                variances,
                &mut rng,
            )
            .unwrap();
            sum[0] += state[0];
            sum[1] += state[1];
            products[0] += state[0] * state[0];
            products[1] += state[0] * state[1];
            products[2] += state[1] * state[0];
            products[3] += state[1] * state[1];
        }
        let mean = [sum[0] / draws as f64, sum[1] / draws as f64];
        let covariance = [
            products[0] / draws as f64 - mean[0] * mean[0],
            products[1] / draws as f64 - mean[0] * mean[1],
            products[2] / draws as f64 - mean[1] * mean[0],
            products[3] / draws as f64 - mean[1] * mean[1],
        ];
        for index in 0..2 {
            assert!((mean[index] - expected_mean[index]).abs() < 0.025);
        }
        for index in 0..4 {
            assert!((covariance[index] - expected_covariance[index]).abs() < 0.035);
        }
    }

    #[test]
    fn terminal_state_stays_finite_at_extreme_covariance_scales() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let state = sample_terminal_state(
            &[f64::NAN; 4],
            [0.0, 0.0],
            [1e200, 0.0, 0.0, 1e200],
            [1e-7, 1e-12, 1.0],
            &mut rng,
        )
        .unwrap();
        assert!(state.iter().all(|value| value.is_finite()));
    }

    /// Filtered mean and covariance after the last observation.
    fn filtered_moments(
        observations: &[f64],
        initial_mean: [f64; 2],
        initial_covariance: [f64; 4],
        [level_variance, slope_variance, observation_variance]: [f64; 3],
    ) -> ([f64; 2], [f64; 4]) {
        let mut mean = initial_mean;
        let mut covariance = StateCovariance::new(initial_covariance).unwrap();
        for &observation in observations {
            let step = kalman_step(
                mean,
                covariance,
                observation,
                level_variance,
                slope_variance,
                observation_variance,
            );
            mean = step.filtered_mean;
            covariance = step.filtered_covariance;
        }
        (mean, covariance.matrix())
    }

    #[test]
    fn filter_matches_exact_arithmetic_at_extreme_scales() {
        // References: the covariance-form recursion in exact rational
        // arithmetic over the stored doubles, with logarithms to 80 digits,
        // rounded once to f64. The log likelihood and the terminal filtered
        // moments are held to 1e-12 relative to their own value.
        let check = |observations: &[f64],
                     initial_mean: [f64; 2],
                     initial_covariance: [f64; 4],
                     variances: [f64; 3],
                     log_likelihood: f64,
                     mean: [f64; 2],
                     [level, cross, slope]: [f64; 3]| {
            let within = |actual: f64, expected: f64, what: &str| {
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs(),
                    "{what}: {actual} vs {expected}"
                );
            };
            let actual =
                log_marginal_likelihood(observations, initial_mean, initial_covariance, variances);
            within(actual, log_likelihood, "log likelihood");
            let (actual_mean, actual_covariance) =
                filtered_moments(observations, initial_mean, initial_covariance, variances);
            within(actual_mean[0], mean[0], "level mean");
            within(actual_mean[1], mean[1], "slope mean");
            within(actual_covariance[0], level, "level variance");
            within(actual_covariance[1], cross, "covariance");
            within(actual_covariance[3], slope, "slope variance");
        };

        // A diffuse prior. Forming the covariances lost the slope variance to
        // cancellation: 1.0 instead of 0.606 after the second observation,
        // and a log likelihood of -41.7417 instead of -41.5781 after five.
        let diffuse = [1.6e16, 0.0, 0.0, 4e15];
        let variances = [0.001, 0.005, 0.3];
        let (_, covariance) = filtered_moments(&[0.2, 0.5], [0.0, 0.0], diffuse, variances);
        assert!((covariance[3] - 0.605_999_999_999_999_9).abs() <= 1e-12 * 0.606);
        check(
            &[0.2, 0.5, 0.9, 1.6, 2.4],
            [0.0, 0.0],
            diffuse,
            variances,
            -41.578_134_755_573_7,
            [2.224_074_003_817_692, 0.555_295_936_142_006_8],
            [
                0.181_997_498_933_898_72,
                0.062_763_917_468_787_65,
                0.039_835_298_566_320_385,
            ],
        );
        // Correlated and diffuse, with gaps.
        check(
            &[0.2, 0.5, f64::NAN, 0.9, 1.6, f64::NAN, 2.4, 3.1],
            [0.3, -0.2],
            [1e14, 3e13, 3e13, 5e13],
            [0.02, 1e-4, 0.05],
            -35.974_651_551_754_924,
            [2.944_282_898_627_642_4, 0.408_958_239_577_601_3],
            [
                0.030_600_620_342_382_47,
                0.005_190_843_558_056_528,
                0.004_600_827_976_011_987,
            ],
        );
        // A level variance 32 orders of magnitude below a correlated slope's.
        check(
            &[0.2, 0.5, 0.9, 1.6, 2.4],
            [0.0, 0.0],
            [1.0, 1e16, 1e16, 2e32],
            variances,
            -42.225_379_782_673_66,
            [2.152_791_316_733_353_4, 0.502_916_908_877_962_8],
            [
                0.166_094_092_689_349_35,
                0.051_077_981_046_541_87,
                0.031_248_389_072_539_802,
            ],
        );
        // State variances 39 orders of magnitude above the observation's.
        check(
            &[0.2, 0.5, 0.9],
            [0.0, 0.0],
            [1.4, 0.25, 0.25, 0.8],
            [2e39, 2e39, 0.3],
            -139.302_483_266_822_66,
            [0.9, 0.3],
            [0.3, 0.18, 3.2e39],
        );
    }

    #[test]
    fn filter_keeps_moments_whose_intermediate_products_leave_the_range() {
        // As above, at scales where some product of two moments leaves the
        // range of normal doubles although every moment is inside it:
        // 1e-100 / 1e300 and 1e-200 * 1e-200 underflow, and (1 + 1e160)^2 and
        // 1e200 * 1e200 overflow.
        let check = |observations: &[f64],
                     initial_covariance: [f64; 4],
                     variances: [f64; 3],
                     log_likelihood: f64,
                     mean: [f64; 2],
                     covariance: [f64; 3]| {
            let within = |actual: f64, expected: f64, what: &str| {
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs(),
                    "{what}: {actual} vs {expected}"
                );
            };
            within(
                log_marginal_likelihood(observations, [0.0, 0.0], initial_covariance, variances),
                log_likelihood,
                "log likelihood",
            );
            let (actual_mean, actual_covariance) =
                filtered_moments(observations, [0.0, 0.0], initial_covariance, variances);
            within(actual_mean[0], mean[0], "level mean");
            within(actual_mean[1], mean[1], "slope mean");
            within(actual_covariance[0], covariance[0], "level variance");
            within(actual_covariance[1], covariance[1], "covariance");
            within(actual_covariance[3], covariance[2], "slope variance");
        };
        check(
            &[0.0, 1e-49, 2e-49],
            [1e300, 0.0, 0.0, 1e-100],
            [1e-101, 1e-110, 1e-100],
            -151.275_536_874_383_36,
            [1.677_419_354_869_927e-49, 6.451_612_903_673_256e-50],
            [
                6.774_193_548_595_213e-101,
                3.225_806_451_945_89e-101,
                3.548_387_098_359_001_4e-101,
            ],
        );
        check(
            &[0.0, 0.0, 0.0],
            [1e-200, 1e-40, 1e-40, 2e120],
            [1e120, 1e120, 1e120],
            -419.652_038_540_723_1,
            [0.0, 0.0],
            [
                8.217_054_263_565_892e119,
                4.186_046_511_627_907e119,
                1.930_232_558_139_534_8e120,
            ],
        );
        check(
            &[0.0, 1e-100, 2e-100],
            [1e-200, 0.0, 0.0, 1e-200],
            [1e-200, 1e-200, 1e-200],
            685.506_289_540_285_8,
            [1.855_855_855_855_855_8e-100, 7.387_387_387_387_387e-101],
            [
                8.198_198_198_198_198e-201,
                4.234_234_234_234_234e-201,
                1.954_954_954_954_955e-200,
            ],
        );
        check(
            &[0.0, 1e100, 2e100],
            [1e200, 0.0, 0.0, 1e200],
            [1e200, 1e200, 1e200],
            -696.044_766_256_141_5,
            [1.855_855_855_855_856e100, 7.387_387_387_387_387e99],
            [
                8.198_198_198_198_197e199,
                4.234_234_234_234_234e199,
                1.954_954_954_954_955e200,
            ],
        );
    }

    #[test]
    fn nearly_singular_initial_covariances_keep_their_conditional_variance() {
        // As above. The first two initial covariances are within rounding of
        // singular, where `slope - cross * (cross / level)` was 60% off in the
        // first and zero in the second; the fourth is too, at a scale of
        // 1e-200. The third has a predicted level/slope covariance of exactly
        // zero at a scale where its square underflows.
        let within = |actual: f64, expected: f64, what: &str| {
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs(),
                "{what}: {actual} vs {expected}"
            );
        };
        let near = [0.1, 0.3, 0.3, 0.9];
        let observations = [0.0, 1e-8, 2e-8];
        within(
            log_marginal_likelihood(&observations, [0.0, 0.0], near, [1e-18; 3]),
            32.172_911_894_631_33,
            "log likelihood",
        );
        let (mean, covariance) = filtered_moments(&observations, [0.0, 0.0], near, [1e-18; 3]);
        within(mean[0], 1.958_740_519_677_019e-8, "level mean");
        within(mean[1], 9.131_379_361_621_454e-9, "slope mean");
        within(covariance[0], 8.671_130_520_517_601e-19, "level variance");
        within(covariance[1], 5.097_116_885_300_212e-19, "covariance");
        within(covariance[3], 2.125_708_817_957_939_3e-18, "slope variance");

        let ordinary = [0.3, -0.7, -0.7, 1.633_333_333_333_333_3];
        within(
            log_marginal_likelihood(&[0.0, 1.0, 2.0], [0.0, 0.0], ordinary, [0.1; 3]),
            -3.056_691_072_876_306,
            "log likelihood",
        );

        let tiny = [2e-200, -1e-200, -1e-200, 1e-200];
        within(
            log_marginal_likelihood(&[0.0; 3], [0.0, 0.0], tiny, [1e-200, 1e-200, 1.0]),
            -2.756_815_599_614_018,
            "log likelihood",
        );
        let (_, covariance) = filtered_moments(&[0.0; 3], [0.0, 0.0], tiny, [1e-200, 1e-200, 1.0]);
        within(covariance[0], 1.3e-199, "level variance");
        within(covariance[1], 5e-200, "covariance");
        within(covariance[3], 4e-200, "slope variance");

        let small = [1e-200, 9e-200, 9e-200, 8.100_000_000_000_001e-199];
        let observations = [0.0, 1e-107, 2e-107];
        within(
            log_marginal_likelihood(&observations, [0.0, 0.0], small, [1e-217; 3]),
            645.816_834_999_266,
            "log likelihood",
        );
        let (mean, covariance) = filtered_moments(&observations, [0.0, 0.0], small, [1e-217; 3]);
        within(mean[0], 1.916_145_444_730_270_6e-107, "level mean");
        within(mean[1], 8.186_003_498_246_669e-108, "slope mean");
        within(covariance[0], 8.432_347_421_309_251e-218, "level variance");
        within(covariance[1], 4.567_935_237_934_298e-218, "covariance");
        within(
            covariance[3],
            2.008_573_745_349_052_3e-217,
            "slope variance",
        );
    }

    #[test]
    fn singular_and_indefinite_initial_covariances_are_rejected() {
        let tiny = f64::from_bits(1); // the smallest positive subnormal
                                      // Determinant zero or negative over the stored doubles, several within
                                      // rounding of zero.
        for covariance in [
            [1.0, 1.0, 1.0, 1.0],
            [25.0, 205.0, 205.0, 1681.0],
            [
                28.912_952_978_830_806,
                10.119_949_964_411_749,
                10.119_949_964_411_749,
                3.542_128_241_179_009,
            ],
            [3.0, 3.872_983_346_207_417, 3.872_983_346_207_417, 5.0],
            [3e-200, 5.5e-199, 5.5e-199, 1.008_333_333_333_333_4e-197],
            [1e-320, 1e-320, 1e-320, 1e-320],
            [2.0 * tiny, 3.0 * tiny, 3.0 * tiny, 4.0 * tiny],
            [1.0, 2.0, 2.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
        ] {
            let error = validate_covariance(covariance).unwrap_err();
            assert!(
                error.to_string().contains(NOT_POSITIVE_DEFINITE),
                "{covariance:?}: {error}"
            );
        }
        // Determinant positive, all but the last within rounding of zero.
        for covariance in [
            [0.3, -0.7, -0.7, 1.633_333_333_333_333_3],
            [1.0, 1.732_050_807_568_877_2, 1.732_050_807_568_877_2, 3.0],
            [0.1, 0.3, 0.3, 0.9],
            [
                1e-200,
                6.999_999_999_999_999e-200,
                6.999_999_999_999_999e-200,
                4.899_999_999_999_999e-199,
            ],
            [0.3, 1.0, 1.0, 3.333_333_333_333_333_5],
            [1.0, 0.999_999_999, 0.999_999_999, 1.0],
        ] {
            assert!(validate_covariance(covariance).is_ok(), "{covariance:?}");
        }
    }

    #[test]
    fn scaled_products_survive_intermediate_overflow_and_underflow() {
        let relative = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 1e-13 * expected.abs(),
                "{actual} vs {expected}"
            );
        };
        // In range throughout: the plain product, exactly.
        assert_eq!(scaled_product(&[-2.0, 3.0], &[4.0]), -1.5);
        // An intermediate subnormal, 1e-320, on the way to 1e-120.
        relative(scaled_product(&[1e-200, 1e-120], &[1.0, 1e-200]), 1e-120);
        // An intermediate overflow, 1e500, on the way to 1e-100, with a sign.
        relative(
            scaled_product(&[-1e200, 1e200], &[1e-100, 1e300, 1e300]),
            -1e-100,
        );
        // A zero numerator is zero whatever the denominators' product does.
        assert_eq!(scaled_product(&[0.0, 1e-200], &[1e-200, 1e-200]), 0.0);
        assert!(scaled_product(&[1.0], &[0.0]).is_infinite());
        assert!(scaled_product(&[f64::NAN], &[1.0]).is_nan());
    }

    #[test]
    fn posterior_means_match_independent_quadrature() {
        // The posterior of the three variances, integrated on a grid in log
        // coordinates with the general state-space filter's likelihood, which
        // the sampler does not use. The terminal level's reference is the same
        // grid's average of the filtered terminal mean.
        let mut observations = simulate_trend([0.3, 0.05, 0.2], 40, 5);
        observations[17] = f64::NAN;
        let initial_mean = [0.0, 0.0];
        let initial_covariance = [4.0, 0.0, 0.0, 1.0];
        // Prior means of 0.05, 0.005 and 1.0 against truths of 0.3, 0.05 and 0.2.
        let priors = [
            InverseGammaPrior {
                shape: 3.0,
                scale: 0.1,
            },
            InverseGammaPrior {
                shape: 3.0,
                scale: 0.01,
            },
            InverseGammaPrior {
                shape: 3.0,
                scale: 2.0,
            },
        ];
        // 40 points a side give means within 1e-5 of 80 points a side, far
        // inside the sampler's Monte Carlo error.
        let points = 40;
        let (lowest, highest) = (-11.0, 2.0);
        let log_grid: Vec<f64> = (0..points)
            .map(|k| lowest + (highest - lowest) * k as f64 / (points - 1) as f64)
            .collect();
        let mut log_weights = Vec::with_capacity(points * points * points);
        for &a in &log_grid {
            for &b in &log_grid {
                for &c in &log_grid {
                    let variances = [a.exp(), b.exp(), c.exp()];
                    let filter = general_trend_model(initial_mean, initial_covariance, variances)
                        .filter(&observations)
                        .unwrap();
                    // The inverse-gamma density times the Jacobian q of log
                    // coordinates, written out rather than shared with the sampler.
                    let log_prior: f64 = priors
                        .iter()
                        .zip(variances)
                        .map(|(prior, q)| -prior.shape * q.ln() - prior.scale / q)
                        .sum();
                    log_weights.push((
                        log_prior + filter.log_likelihood,
                        variances,
                        filter.filtered_means.last().unwrap()[0],
                    ));
                }
            }
        }
        let peak = log_weights
            .iter()
            .map(|(w, _, _)| *w)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        let mut reference = [0.0; 4];
        for (log_weight, variances, terminal_level) in &log_weights {
            let weight = (log_weight - peak).exp();
            total += weight;
            for index in 0..3 {
                reference[index] += weight * variances[index];
            }
            reference[3] += weight * terminal_level;
        }
        reference.iter_mut().for_each(|value| *value /= total);

        let mut fit_config = config(31);
        fit_config.initial_mean = initial_mean;
        fit_config.initial_covariance = initial_covariance;
        [
            fit_config.level_variance_prior,
            fit_config.slope_variance_prior,
            fit_config.observation_variance_prior,
        ] = priors;
        fit_config.num_chains = 4;
        fit_config.num_warmup = 500;
        fit_config.num_draws = 5000;
        let report = fit_bayesian_local_linear_trend(&observations, &fit_config)
            .unwrap()
            .diagnostics();
        for (parameter, expected) in report.params.iter().zip(reference) {
            assert!(
                (parameter.mean - expected).abs() < 4.0 * parameter.mcse_mean,
                "{}: sampled mean {} (MCSE {}) vs quadrature {expected}",
                parameter.name,
                parameter.mean,
                parameter.mcse_mean,
            );
        }
        // Negative control: the data must move every variance's mean well
        // away from its prior mean, scale / (shape - 1), or matching the
        // quadrature would not show the likelihood is used.
        for ((parameter, expected), prior) in report.params.iter().zip(reference).zip(priors) {
            let prior_mean = prior.scale / (prior.shape - 1.0);
            assert!(
                (prior_mean - expected).abs() > 8.0 * parameter.mcse_mean,
                "{}: prior mean {prior_mean} is too close to the posterior mean {expected}",
                parameter.name,
            );
        }
    }

    /// Simulate `count` observations from a local linear trend that starts at
    /// level 0 with slope 0.02.
    fn simulate_trend(variances: [f64; 3], count: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let [level_sd, slope_sd, observation_sd] = variances.map(f64::sqrt);
        let (mut level, mut slope) = (0.0, 0.02);
        (0..count)
            .map(|_| {
                level += slope + standard_normal(&mut rng) * level_sd;
                slope += standard_normal(&mut rng) * slope_sd;
                level + standard_normal(&mut rng) * observation_sd
            })
            .collect()
    }

    #[test]
    fn state_variances_mix_in_every_signal_to_noise_regime() {
        // The sampler this replaced, FFBS followed by conjugate draws of each
        // variance given the states, gave the level variance a split R-hat of
        // 1.20 to 1.75 and a bulk ESS of 6 to 15 out of 2000 draws on every one
        // of these twelve series, and the slope variance an R-hat up to 1.16
        // and a bulk ESS of 17 to 106. Interweaving a non-centred update into
        // it (ASIS) raised the level variance's bulk ESS only to 13 to 181.
        // The thresholds are Vehtari et al. (2021)'s: R-hat below 1.01 is
        // their target, and 1.02 leaves room for its sampling noise over 36
        // checks; 400 bulk and tail ESS is their minimum of 100 per chain.
        let regimes = [
            ("balanced", [0.05, 0.005, 0.3]),
            ("near-zero slope variance", [0.05, 1e-6, 0.3]),
            ("near-zero level variance", [1e-6, 0.002, 0.3]),
            ("low signal-to-noise", [0.005, 0.0005, 1.0]),
        ];
        for (regime, variances) in regimes {
            for seed in 0..3 {
                let observations = simulate_trend(variances, 200, 7_000 + seed);
                let mut fit_config = config(100 + seed);
                fit_config.initial_mean = [observations[0], 0.0];
                fit_config.initial_covariance = [10.0, 0.0, 0.0, 1.0];
                fit_config.level_variance_prior = InverseGammaPrior {
                    shape: 2.0,
                    scale: 0.01,
                };
                fit_config.slope_variance_prior = InverseGammaPrior {
                    shape: 2.0,
                    scale: 0.001,
                };
                fit_config.observation_variance_prior = InverseGammaPrior {
                    shape: 2.0,
                    scale: 0.1,
                };
                fit_config.num_chains = 4;
                fit_config.num_warmup = 300;
                fit_config.num_draws = 500;
                let report = fit_bayesian_local_linear_trend(&observations, &fit_config)
                    .unwrap()
                    .diagnostics();
                for parameter in &report.params[..3] {
                    assert!(
                        parameter.r_hat < 1.02
                            && parameter.ess_bulk >= 400.0
                            && parameter.ess_tail >= 400.0,
                        "{regime}, seed {seed}: {} has R-hat {:.3}, bulk ESS {:.0}, \
                         tail ESS {:.0}",
                        parameter.name,
                        parameter.r_hat,
                        parameter.ess_bulk,
                        parameter.ess_tail
                    );
                }
            }
        }
    }

    #[test]
    fn recovers_signal_and_produces_widening_forecasts() {
        let true_level_variance: f64 = 0.12;
        let true_slope_variance: f64 = 0.025;
        let true_observation_variance: f64 = 0.35;
        let mut rng = ChaCha8Rng::seed_from_u64(810);
        let mut level = 0.0;
        let mut slope = 0.08;
        // 280 points used to be enough only because config()'s priors already carried
        // the answer. With a prior that does not, the level/observation split at that
        // length is not identified: the posterior mean of the level variance comes out
        // near 0.03 against a truth of 0.12. It is identified at 2000.
        let count = 2000;
        let mut observations = Vec::with_capacity(count);
        for _ in 0..count {
            level += slope + standard_normal(&mut rng) * true_level_variance.sqrt();
            slope += standard_normal(&mut rng) * true_slope_variance.sqrt();
            observations.push(level + standard_normal(&mut rng) * true_observation_variance.sqrt());
        }
        observations[70] = f64::NAN;
        observations[171] = f64::NAN;

        let mut fit_config = config(811);
        fit_config.num_warmup = 500;
        fit_config.num_draws = 500;
        fit_config.thinning = 2;
        // config()'s own priors have means 0.25/2 = 0.125, 0.05/2 = 0.025 and
        // 0.5/2 = 0.25, against truths of 0.12, 0.025 and 0.35. The first is 0.005 from
        // its truth and the second is exactly equal to it, so under the tolerances this
        // test used to carry, prior draws satisfied all three assertions: it asserted
        // nothing about the observations. Fit with priors whose means are 0.025, 0.0025
        // and 0.025 instead, a factor of 4.8, 10 and 14 below the three truths. The
        // negative control below keeps them outside the accepted windows.
        fit_config.level_variance_prior = InverseGammaPrior {
            shape: 3.0,
            scale: 0.05,
        };
        fit_config.slope_variance_prior = InverseGammaPrior {
            shape: 3.0,
            scale: 0.005,
        };
        fit_config.observation_variance_prior = InverseGammaPrior {
            shape: 3.0,
            scale: 0.05,
        };
        let posterior = fit_bayesian_local_linear_trend(&observations, &fit_config).unwrap();
        let count = posterior.chains.iter().map(Vec::len).sum::<usize>() as f64;
        let means = posterior
            .chains
            .iter()
            .flatten()
            .fold([0.0; 3], |mut sums, draw| {
                sums[0] += draw.level_variance;
                sums[1] += draw.slope_variance;
                sums[2] += draw.observation_variance;
                sums
            });
        let tolerances = [0.045, 0.008, 0.06];
        let truths = [
            true_level_variance,
            true_slope_variance,
            true_observation_variance,
        ];
        for index in 0..3 {
            assert!(
                (means[index] / count - truths[index]).abs() < tolerances[index],
                "parameter {index}: {} vs {}",
                means[index] / count,
                truths[index]
            );
        }

        // Negative control: a sampler that ignored the observations would report the
        // prior means, scale / (shape - 1) = 0.025, 0.0025 and 0.025. Every one must
        // sit outside the window just asserted, or those assertions prove nothing.
        let priors = [
            fit_config.level_variance_prior,
            fit_config.slope_variance_prior,
            fit_config.observation_variance_prior,
        ];
        for index in 0..3 {
            let prior_mean = priors[index].scale / (priors[index].shape - 1.0);
            // One tolerance-width of clearance, not merely exclusion: a window
            // whose edge sits just short of the prior mean demonstrates nothing,
            // and `> tolerances[index]` alone permits exactly that.
            assert!(
                (prior_mean - truths[index]).abs() > 2.0 * tolerances[index],
                "parameter {index}: prior mean {prior_mean} is within one tolerance-width \
                 of the window around {}",
                truths[index]
            );
        }

        let forecast = posterior.forecast(15, 812).unwrap();
        let intervals = forecast.observation_quantiles(&[0.1, 0.9]).unwrap();
        let first_width = intervals[1].values[0] - intervals[0].values[0];
        let last_width = intervals[1].values[14] - intervals[0].values[14];
        assert!(last_width > first_width, "{first_width} vs {last_width}");
    }
}
