//! Seeded Bayesian inference and posterior prediction for a local-level model.
//!
//! This module uses a conjugate Gibbs sampler with forward-filtering,
//! backward-sampling (FFBS). The model is
//!
//! ```text
//! level[-1] ~ Normal(initial_mean, initial_variance)
//! level[t] = level[t - 1] + Normal(0, process_variance)
//! y[t] = level[t] + Normal(0, observation_variance)
//! ```
//!
//! Both variances have caller-specified inverse-gamma priors; their scale is
//! in squared observation units, so there is deliberately no universal
//! default. Missing observations may be represented by `NaN`; their time
//! positions are retained, while infinities are rejected.
//!
//! Independent chains execute on the active Rayon pool (or its global pool),
//! while indexed collection preserves deterministic chain ordering.

use crate::seeding::chain_seed;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;
use std::error::Error;
use std::fmt;

type LocalLevelChainPaths = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Inverse-gamma prior parameterized by shape and scale.
///
/// Its density is proportional to `x^(-shape - 1) * exp(-scale / x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGammaPrior {
    pub shape: f64,
    pub scale: f64,
}

impl InverseGammaPrior {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesianForecastError> {
        let prior = Self { shape, scale };
        prior.validate("inverse-gamma")?;
        Ok(prior)
    }

    fn validate(&self, name: &str) -> Result<(), BayesianForecastError> {
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(BayesianForecastError::InvalidConfiguration(format!(
                "{name} shape must be finite and strictly positive"
            )));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(BayesianForecastError::InvalidConfiguration(format!(
                "{name} scale must be finite and strictly positive"
            )));
        }
        Ok(())
    }

    fn mode(self) -> f64 {
        self.scale / (self.shape + 1.0)
    }
}

#[derive(Debug, Clone)]
pub struct BayesianLocalLevelConfig {
    pub initial_mean: f64,
    pub initial_variance: f64,
    pub process_variance_prior: InverseGammaPrior,
    pub observation_variance_prior: InverseGammaPrior,
    pub num_chains: usize,
    pub num_warmup: usize,
    pub num_draws: usize,
    pub thinning: usize,
    pub seed: u64,
}

impl BayesianLocalLevelConfig {
    fn validate(&self) -> Result<(), BayesianForecastError> {
        if !self.initial_mean.is_finite() {
            return Err(BayesianForecastError::InvalidConfiguration(
                "initial mean must be finite".into(),
            ));
        }
        if !self.initial_variance.is_finite() || self.initial_variance <= 0.0 {
            return Err(BayesianForecastError::InvalidConfiguration(
                "initial variance must be finite and strictly positive".into(),
            ));
        }
        self.process_variance_prior
            .validate("process-variance prior")?;
        self.observation_variance_prior
            .validate("observation-variance prior")?;
        if self.num_chains == 0 {
            return Err(BayesianForecastError::InvalidConfiguration(
                "number of chains must be positive".into(),
            ));
        }
        if self.num_draws == 0 {
            return Err(BayesianForecastError::InvalidConfiguration(
                "number of posterior draws must be positive".into(),
            ));
        }
        if self.thinning == 0 {
            return Err(BayesianForecastError::InvalidConfiguration(
                "thinning must be positive".into(),
            ));
        }
        self.num_draws
            .checked_mul(self.thinning)
            .and_then(|saved_iterations| self.num_warmup.checked_add(saved_iterations))
            .ok_or_else(|| {
                BayesianForecastError::InvalidConfiguration(
                    "warmup, draws, and thinning imply too many iterations".into(),
                )
            })?;
        Ok(())
    }
}

/// One joint posterior draw of variance parameters and the terminal level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelPosteriorDraw {
    pub process_variance: f64,
    pub observation_variance: f64,
    pub terminal_level: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalLevelPosterior {
    /// Joint posterior samples indexed as `[chain][draw]`.
    pub chains: Vec<Vec<LocalLevelPosteriorDraw>>,
}

impl LocalLevelPosterior {
    /// Draw future observations, pairing each posterior draw with one path.
    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<PosteriorPredictiveForecast, BayesianForecastError> {
        if horizon == 0 {
            return Err(BayesianForecastError::InvalidConfiguration(
                "forecast horizon must be positive".into(),
            ));
        }
        if self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(BayesianForecastError::InvalidConfiguration(
                "every posterior chain must contain at least one draw".into(),
            ));
        }

        let chain_paths: Vec<LocalLevelChainPaths> = self
            .chains
            .par_iter()
            .enumerate()
            .map(|(chain_index, posterior_chain)| {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(chain_seed(seed, chain_index, FORECAST_SEED_DOMAIN));
                let mut chain_state_paths = Vec::with_capacity(posterior_chain.len());
                let mut chain_observation_paths = Vec::with_capacity(posterior_chain.len());
                for draw in posterior_chain {
                    validate_positive_variance("process", draw.process_variance)?;
                    validate_positive_variance("observation", draw.observation_variance)?;
                    if !draw.terminal_level.is_finite() {
                        return Err(BayesianForecastError::NumericalFailure(
                            "posterior terminal level is not finite".into(),
                        ));
                    }

                    let process_sd = draw.process_variance.sqrt();
                    let observation_sd = draw.observation_variance.sqrt();
                    let mut level = draw.terminal_level;
                    let mut state_path = Vec::with_capacity(horizon);
                    let mut observation_path = Vec::with_capacity(horizon);
                    for _ in 0..horizon {
                        level += standard_normal(&mut rng) * process_sd;
                        let observation = level + standard_normal(&mut rng) * observation_sd;
                        if !level.is_finite() || !observation.is_finite() {
                            return Err(BayesianForecastError::NumericalFailure(
                                "posterior predictive simulation overflowed".into(),
                            ));
                        }
                        state_path.push(level);
                        observation_path.push(observation);
                    }
                    chain_state_paths.push(state_path);
                    chain_observation_paths.push(observation_path);
                }
                Ok((chain_state_paths, chain_observation_paths))
            })
            .collect::<Result<_, BayesianForecastError>>()?;

        let mut state_paths = Vec::with_capacity(chain_paths.len());
        let mut observation_paths = Vec::with_capacity(chain_paths.len());
        for (chain_state_paths, chain_observation_paths) in chain_paths {
            state_paths.push(chain_state_paths);
            observation_paths.push(chain_observation_paths);
        }

        Ok(PosteriorPredictiveForecast {
            state_paths,
            observation_paths,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastQuantile {
    pub probability: f64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorPredictiveForecast {
    /// Coherent latent-level paths indexed as `[chain][draw][forecast_step]`.
    pub state_paths: Vec<Vec<Vec<f64>>>,
    /// Coherent observation paths indexed as `[chain][draw][forecast_step]`.
    pub observation_paths: Vec<Vec<Vec<f64>>>,
}

impl PosteriorPredictiveForecast {
    pub fn horizon(&self) -> usize {
        self.observation_paths
            .first()
            .and_then(|chain| chain.first())
            .map_or(0, Vec::len)
    }

    pub fn state_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.state_paths)
    }

    pub fn observation_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.observation_paths)
    }

    /// Compute empirical latent-state quantiles at every horizon.
    pub fn state_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.state_paths, probabilities)
    }

    /// Compute empirical posterior-predictive observation quantiles.
    pub fn observation_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.observation_paths, probabilities)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BayesianForecastError {
    InvalidConfiguration(String),
    InvalidObservations(String),
    NumericalFailure(String),
}

impl fmt::Display for BayesianForecastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => write!(f, "invalid configuration: {message}"),
            Self::InvalidObservations(message) => write!(f, "invalid observations: {message}"),
            Self::NumericalFailure(message) => write!(f, "numerical failure: {message}"),
        }
    }
}

impl Error for BayesianForecastError {}

/// Fit the local-level model with a seeded conjugate Gibbs sampler.
pub fn fit_bayesian_local_level(
    observations: &[f64],
    config: &BayesianLocalLevelConfig,
) -> Result<LocalLevelPosterior, BayesianForecastError> {
    config.validate()?;
    validate_observations(observations)?;

    let total_iterations = config.num_warmup + config.num_draws * config.thinning;
    let observed_count = observations.iter().filter(|value| !value.is_nan()).count();
    let chains = (0..config.num_chains)
        .into_par_iter()
        .map(|chain_index| {
            let mut rng =
                ChaCha8Rng::seed_from_u64(chain_seed(config.seed, chain_index, FIT_SEED_DOMAIN));
            let mut process_variance = config.process_variance_prior.mode();
            let mut observation_variance = config.observation_variance_prior.mode();
            let mut posterior_draws = Vec::with_capacity(config.num_draws);

            for iteration in 0..total_iterations {
                // Includes x[-1] followed by x[0] through x[T - 1].
                let levels = sample_levels_ffbs(
                    observations,
                    config.initial_mean,
                    config.initial_variance,
                    process_variance,
                    observation_variance,
                    &mut rng,
                )?;

                let process_sum_sq: f64 = levels
                    .windows(2)
                    .map(|pair| {
                        let difference = pair[1] - pair[0];
                        difference * difference
                    })
                    .sum();
                process_variance = sample_inverse_gamma(
                    config.process_variance_prior.shape + observations.len() as f64 / 2.0,
                    config.process_variance_prior.scale + process_sum_sq / 2.0,
                    &mut rng,
                )?;

                let observation_sum_sq: f64 = observations
                    .iter()
                    .zip(&levels[1..])
                    .filter(|(observation, _)| !observation.is_nan())
                    .map(|(observation, level)| {
                        let residual = observation - level;
                        residual * residual
                    })
                    .sum();
                observation_variance = sample_inverse_gamma(
                    config.observation_variance_prior.shape + observed_count as f64 / 2.0,
                    config.observation_variance_prior.scale + observation_sum_sq / 2.0,
                    &mut rng,
                )?;

                if iteration >= config.num_warmup
                    && (iteration + 1 - config.num_warmup).is_multiple_of(config.thinning)
                {
                    posterior_draws.push(LocalLevelPosteriorDraw {
                        process_variance,
                        observation_variance,
                        terminal_level: *levels.last().expect("observations are non-empty"),
                    });
                }
            }
            debug_assert_eq!(posterior_draws.len(), config.num_draws);
            Ok(posterior_draws)
        })
        .collect::<Result<Vec<_>, BayesianForecastError>>()?;

    Ok(LocalLevelPosterior { chains })
}

fn validate_observations(observations: &[f64]) -> Result<(), BayesianForecastError> {
    if observations.is_empty() {
        return Err(BayesianForecastError::InvalidObservations(
            "at least one observation is required".into(),
        ));
    }
    if observations.iter().any(|value| value.is_infinite()) {
        return Err(BayesianForecastError::InvalidObservations(
            "observations may be finite or NaN, but not infinite".into(),
        ));
    }
    if observations.iter().filter(|value| !value.is_nan()).count() < 2 {
        return Err(BayesianForecastError::InvalidObservations(
            "at least two finite observations are required".into(),
        ));
    }
    Ok(())
}

/// One forward-filtering pass of the local-level Kalman recursion.
///
/// Index zero holds the pre-transition state `x[-1]`; index `time + 1` holds
/// the state filtered on observations up to and including `time`.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalLevelFilter {
    pub filtered_means: Vec<f64>,
    pub filtered_variances: Vec<f64>,
}

/// `a b / (a + b)` for two positive variances, evaluated so that no
/// intermediate leaves the range when the answer does not.
///
/// The three orderings are not interchangeable in floating point. `a * b`
/// first overflows once `a b > f64::MAX` and underflows once `a b` reaches the
/// subnormals, neither of which is a limit on the result: at `a = 6e-162,
/// b = 3e-162` the product is subnormal and the answer comes back 9.8% high,
/// and below about `1e-170` it underflows to zero, so the positivity guard
/// rejects a perfectly well scaled filtering problem. Dividing first is no
/// better in the other direction, because `a / (a + b)` itself underflows
/// once `a / b` falls below about `1e-308`: at `a = 1e-200, b = 1e120` that
/// costs eleven significant digits, and at `b = 1e200` it returns zero.
///
/// Dividing by the sum whichever factor is the larger keeps that quotient in
/// `[0.5, 1)`, so the surviving product lies in `[min(a, b) / 2, min(a, b))`
/// and cannot leave the range in either direction. Over 400k log-uniform pairs
/// spanning `1e-300` to `1e300` this form is within two ulp of the correctly
/// rounded result everywhere and never returns zero or an infinity, where the
/// two single-ordering forms are wrong by up to 8e15 ulp and fail outright on
/// about 22% of the pairs for `a * b` first, which loses both ends of the
/// range, and about 11% for dividing first, which loses only the lower one. The scalar collapse of the Joseph
/// form that `state_space.rs` uses, `(1 - k)^2 a + k^2 b`, scores the same two
/// ulp for more arithmetic and without the interval argument.
///
/// `sum` is passed in because both callers have already formed `a + b` for the
/// gain. A non-finite sum is not rescued here: it yields a zero or non-finite
/// result, which the caller's own positivity check rejects.
fn harmonic_half(a: f64, b: f64, sum: f64) -> f64 {
    if a <= b {
        a * (b / sum)
    } else {
        b * (a / sum)
    }
}

/// Run the forward-filtering pass that the Gibbs sampler's FFBS step uses.
///
/// Exposed because the variance recursion is the numerically delicate half of
/// FFBS and the half a caller can pin exactly against a closed form: it is a
/// deterministic function of the three variances and of which observations are
/// present, with no dependence on the RNG.
///
/// This validates the variances and the initial mean, but takes `observations`
/// as given: an infinite observation is reported as a `NumericalFailure` on the
/// filtered level rather than as `InvalidObservations`, which is the check
/// [`fit_bayesian_local_level`] applies at its own boundary.
pub fn filter_local_level(
    observations: &[f64],
    initial_mean: f64,
    initial_variance: f64,
    process_variance: f64,
    observation_variance: f64,
) -> Result<LocalLevelFilter, BayesianForecastError> {
    validate_positive_variance("process", process_variance)?;
    validate_positive_variance("observation", observation_variance)?;
    validate_positive_variance("initial state", initial_variance)?;
    if !initial_mean.is_finite() {
        return Err(BayesianForecastError::NumericalFailure(
            "initial level must be finite".into(),
        ));
    }

    let len = observations.len();
    let mut filtered_means = Vec::with_capacity(len + 1);
    let mut filtered_variances = Vec::with_capacity(len + 1);
    filtered_means.push(initial_mean);
    filtered_variances.push(initial_variance);

    for (time, &observation) in observations.iter().enumerate() {
        let predicted_mean = filtered_means[time];
        let predicted_variance = filtered_variances[time] + process_variance;
        validate_positive_variance("predicted state", predicted_variance)?;

        let (filtered_mean, filtered_variance) = if observation.is_nan() {
            (predicted_mean, predicted_variance)
        } else {
            let innovation_variance = predicted_variance + observation_variance;
            validate_positive_variance("innovation", innovation_variance)?;
            let gain = predicted_variance / innovation_variance;
            let mean = predicted_mean + gain * (observation - predicted_mean);
            let variance = harmonic_half(
                predicted_variance,
                observation_variance,
                innovation_variance,
            );
            (mean, variance)
        };
        if !filtered_mean.is_finite() {
            return Err(BayesianForecastError::NumericalFailure(format!(
                "filtered level at time {time} is not finite"
            )));
        }
        validate_positive_variance("filtered state", filtered_variance)?;
        filtered_means.push(filtered_mean);
        filtered_variances.push(filtered_variance);
    }

    Ok(LocalLevelFilter {
        filtered_means,
        filtered_variances,
    })
}

fn sample_levels_ffbs(
    observations: &[f64],
    initial_mean: f64,
    initial_variance: f64,
    process_variance: f64,
    observation_variance: f64,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>, BayesianForecastError> {
    let LocalLevelFilter {
        filtered_means,
        filtered_variances,
    } = filter_local_level(
        observations,
        initial_mean,
        initial_variance,
        process_variance,
        observation_variance,
    )?;

    let len = observations.len();
    let mut levels = vec![0.0; len + 1];
    levels[len] = sample_normal(filtered_means[len], filtered_variances[len], rng)?;
    for time in (0..len).rev() {
        let filtered_variance = filtered_variances[time];
        let next_prediction_variance = filtered_variance + process_variance;
        let smoothing_gain = filtered_variance / next_prediction_variance;
        let mean =
            filtered_means[time] + smoothing_gain * (levels[time + 1] - filtered_means[time]);
        // The backward conditional variance is the same `a b / (a + b)` the
        // filtering update computes, and needs the same care.
        let variance = harmonic_half(
            filtered_variance,
            process_variance,
            next_prediction_variance,
        );
        levels[time] = sample_normal(mean, variance, rng)?;
    }
    Ok(levels)
}

fn sample_inverse_gamma(
    shape: f64,
    scale: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    if !shape.is_finite() || shape <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        return Err(BayesianForecastError::NumericalFailure(
            "invalid inverse-gamma posterior parameters".into(),
        ));
    }
    let gamma = Gamma::new(shape, 1.0 / scale).map_err(|error| {
        BayesianForecastError::NumericalFailure(format!(
            "could not construct gamma distribution: {error}"
        ))
    })?;
    let precision = gamma.sample(rng);
    let variance = 1.0 / precision;
    validate_positive_variance("sampled", variance)?;
    Ok(variance)
}

fn sample_normal(
    mean: f64,
    variance: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    validate_positive_variance("normal", variance)?;
    let draw = mean + standard_normal(rng) * variance.sqrt();
    if !draw.is_finite() {
        return Err(BayesianForecastError::NumericalFailure(
            "normal simulation produced a non-finite draw".into(),
        ));
    }
    Ok(draw)
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

fn validate_positive_variance(name: &str, variance: f64) -> Result<(), BayesianForecastError> {
    if !variance.is_finite() || variance <= 0.0 {
        return Err(BayesianForecastError::NumericalFailure(format!(
            "{name} variance must be finite and strictly positive"
        )));
    }
    Ok(())
}

const FIT_SEED_DOMAIN: u64 = 0x4649_545F_4C4F_434C;
const FORECAST_SEED_DOMAIN: u64 = 0x4652_4353_545F_4C4C;

/// Posterior-predictive mean at each horizon, from the scale-aware
/// implementation the sampler's `mean()` accessors already use.
///
/// Accumulating the paths and dividing by their count at the end overflows on
/// input that is entirely finite: two paths holding `1e308` sum to infinity,
/// and the infinity survives the division. See
/// [`crate::diagnostics::scaled_moments`], which centres the draws at one
/// horizon on the first of them and divides by the largest deviation from it
/// before summing, so no partial sum can leave the representable range.
///
/// `validate_paths` has already rejected an empty, ragged or non-finite
/// forecast, so the `NaN` that `scaled_moments` reports for those cases cannot
/// reach a caller from here.
fn path_means(paths: &[Vec<Vec<f64>>]) -> Result<Vec<f64>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    let means: Vec<f64> = (0..horizon)
        .map(|step| {
            crate::diagnostics::scaled_moments(|| paths.iter().flatten().map(|path| path[step])).0
        })
        .collect();
    Ok(means)
}

fn path_quantiles(
    paths: &[Vec<Vec<f64>>],
    probabilities: &[f64],
) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    for &probability in probabilities {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(BayesianForecastError::InvalidConfiguration(
                "quantile probabilities must be finite and between zero and one".into(),
            ));
        }
    }

    let mut ordered_by_step = Vec::with_capacity(horizon);
    for step in 0..horizon {
        let mut values: Vec<f64> = paths.iter().flatten().map(|path| path[step]).collect();
        values.sort_by(f64::total_cmp);
        ordered_by_step.push(values);
    }

    Ok(probabilities
        .iter()
        .map(|&probability| ForecastQuantile {
            probability,
            values: ordered_by_step
                .iter()
                .map(|ordered| interpolated_quantile(ordered, probability))
                .collect(),
        })
        .collect())
}

fn validate_paths(paths: &[Vec<Vec<f64>>]) -> Result<usize, BayesianForecastError> {
    let horizon = paths
        .first()
        .and_then(|chain| chain.first())
        .map_or(0, Vec::len);
    if paths.is_empty() || paths.iter().any(Vec::is_empty) || horizon == 0 {
        return Err(BayesianForecastError::InvalidConfiguration(
            "forecast must contain at least one non-empty path per chain".into(),
        ));
    }
    if paths.iter().flatten().any(|path| path.len() != horizon) {
        return Err(BayesianForecastError::InvalidConfiguration(
            "forecast paths must all have the same horizon".into(),
        ));
    }
    if paths
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(BayesianForecastError::NumericalFailure(
            "forecast contains a non-finite value".into(),
        ));
    }
    Ok(horizon)
}

fn interpolated_quantile(ordered: &[f64], probability: f64) -> f64 {
    let index = probability * (ordered.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        ordered[lower]
    } else {
        let weight = index - lower as f64;
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_forecast_seed_domains_are_distinct() {
        assert_ne!(
            chain_seed(42, 0, FIT_SEED_DOMAIN),
            chain_seed(42, 0, FORECAST_SEED_DOMAIN)
        );
    }

    fn compact_config(seed: u64) -> BayesianLocalLevelConfig {
        BayesianLocalLevelConfig {
            initial_mean: 0.0,
            initial_variance: 4.0,
            process_variance_prior: InverseGammaPrior {
                shape: 2.5,
                scale: 0.3,
            },
            observation_variance_prior: InverseGammaPrior {
                shape: 2.5,
                scale: 0.6,
            },
            num_chains: 2,
            num_warmup: 100,
            num_draws: 200,
            thinning: 1,
            seed,
        }
    }

    #[test]
    fn fit_and_forecast_are_bitwise_identical_across_rayon_pool_sizes() {
        let observations = [0.2, -0.1, f64::NAN, 0.4, 0.3, 0.6];
        let config = compact_config(18);
        let run = || {
            let posterior = fit_bayesian_local_level(&observations, &config).unwrap();
            let forecast = posterior.forecast(5, 19).unwrap();
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
    fn posterior_and_forecast_are_seeded_and_quantiles_are_ordered() {
        let observations = [0.2, -0.1, f64::NAN, 0.4, 0.3, 0.6];
        let config = compact_config(17);
        let first = fit_bayesian_local_level(&observations, &config).unwrap();
        let second = fit_bayesian_local_level(&observations, &config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.chains.len(), 2);
        assert!(first.chains.iter().all(|chain| chain.len() == 200));

        let forecast = first.forecast(4, 91).unwrap();
        assert_eq!(forecast, second.forecast(4, 91).unwrap());
        assert_eq!(forecast.state_paths.len(), 2);
        assert!(forecast
            .observation_paths
            .iter()
            .all(|chain| chain.len() == 200));
        assert_eq!(forecast.horizon(), 4);
        for quantiles in [
            forecast.state_quantiles(&[0.05, 0.5, 0.95]).unwrap(),
            forecast.observation_quantiles(&[0.05, 0.5, 0.95]).unwrap(),
        ] {
            for step in 0..4 {
                assert!(quantiles[0].values[step] <= quantiles[1].values[step]);
                assert!(quantiles[1].values[step] <= quantiles[2].values[step]);
            }
        }
    }

    #[test]
    fn invalid_inputs_are_rejected_without_panicking() {
        let config = compact_config(3);
        assert!(fit_bayesian_local_level(&[], &config).is_err());
        assert!(fit_bayesian_local_level(&[f64::NAN], &config).is_err());
        assert!(fit_bayesian_local_level(&[f64::INFINITY], &config).is_err());
        assert!(fit_bayesian_local_level(&[1.0, f64::NAN], &config).is_err());

        let mut invalid = config;
        invalid.thinning = 0;
        assert!(fit_bayesian_local_level(&[1.0, 2.0], &invalid).is_err());
    }

    #[test]
    fn recovers_variance_scale_and_builds_widening_forecasts() {
        let process_variance: f64 = 0.2;
        let observation_variance: f64 = 0.5;
        let mut rng = ChaCha8Rng::seed_from_u64(700);
        let mut level = 0.0;
        let mut observations = Vec::with_capacity(240);
        for _ in 0..240 {
            level += standard_normal(&mut rng) * process_variance.sqrt();
            observations.push(level + standard_normal(&mut rng) * observation_variance.sqrt());
        }

        let mut config = compact_config(701);
        config.initial_variance = 1.0;
        config.num_warmup = 400;
        config.num_draws = 500;
        config.thinning = 2;
        // compact_config's own priors have means 0.3/1.5 = 0.2 and 0.6/1.5 = 0.4. The
        // first is exactly `process_variance` and the second is 0.1 from
        // `observation_variance`, so under the tolerances below this test used to pass
        // on prior draws alone: it asserted nothing about the observations. Fit with a
        // prior deliberately far below both truths instead, so the windows can only be
        // reached by conditioning on the data. The negative control below keeps it so.
        config.process_variance_prior = InverseGammaPrior {
            shape: 3.0,
            scale: 0.05,
        };
        config.observation_variance_prior = config.process_variance_prior;
        let posterior = fit_bayesian_local_level(&observations, &config).unwrap();
        let draw_count = posterior.chains.iter().map(Vec::len).sum::<usize>();
        let mean_process = posterior
            .chains
            .iter()
            .flatten()
            .map(|draw| draw.process_variance)
            .sum::<f64>()
            / draw_count as f64;
        let mean_observation = posterior
            .chains
            .iter()
            .flatten()
            .map(|draw| draw.observation_variance)
            .sum::<f64>()
            / draw_count as f64;
        let process_tolerance = 0.06;
        let observation_tolerance = 0.10;
        assert!(
            (mean_process - process_variance).abs() < process_tolerance,
            "{mean_process}"
        );
        assert!(
            (mean_observation - observation_variance).abs() < observation_tolerance,
            "{mean_observation}"
        );

        // Negative control: a sampler that ignored the observations would report the
        // prior means, scale / (shape - 1) = 0.025 for both. Those must sit outside the
        // windows just asserted, or the two assertions above prove nothing.
        let prior_mean = |prior: InverseGammaPrior| prior.scale / (prior.shape - 1.0);
        assert!(
            (prior_mean(config.process_variance_prior) - process_variance).abs()
                > process_tolerance,
            "the process-variance prior mean is inside the accepted window"
        );
        assert!(
            (prior_mean(config.observation_variance_prior) - observation_variance).abs()
                > observation_tolerance,
            "the observation-variance prior mean is inside the accepted window"
        );

        let forecast = posterior.forecast(12, 702).unwrap();
        let intervals = forecast.observation_quantiles(&[0.1, 0.9]).unwrap();
        let first_width = intervals[1].values[0] - intervals[0].values[0];
        let last_width = intervals[1].values[11] - intervals[0].values[11];
        assert!(last_width > first_width, "{first_width} vs {last_width}");
    }

    #[test]
    fn ffbs_matches_pre_transition_initial_state_and_preserves_trailing_missing_steps() {
        use crate::state_space::LinearGaussianStateSpace;

        let observations = [0.3, -0.2, f64::NAN, f64::NAN];
        let process_variance = 0.4;
        let observation_variance = 0.7;
        let initial_mean = -0.1;
        let initial_variance = 1.2;
        let filter = LinearGaussianStateSpace::local_level(
            process_variance,
            observation_variance,
            initial_mean,
            initial_variance,
        )
        .unwrap()
        .filter(&observations)
        .unwrap();
        let expected_mean = filter.filtered_means.last().unwrap()[0];
        let expected_variance = filter.filtered_covariances.last().unwrap()[0];

        let mut rng = ChaCha8Rng::seed_from_u64(88);
        let draws = 20_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..draws {
            let levels = sample_levels_ffbs(
                &observations,
                initial_mean,
                initial_variance,
                process_variance,
                observation_variance,
                &mut rng,
            )
            .unwrap();
            assert_eq!(levels.len(), observations.len() + 1);
            let terminal = *levels.last().unwrap();
            sum += terminal;
            sum_sq += terminal * terminal;
        }
        let sampled_mean = sum / draws as f64;
        let sampled_variance = sum_sq / draws as f64 - sampled_mean * sampled_mean;
        assert!((sampled_mean - expected_mean).abs() < 0.03);
        assert!((sampled_variance - expected_variance).abs() < 0.04);
    }
}
