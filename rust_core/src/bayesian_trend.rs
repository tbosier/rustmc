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
//! posterior is sampled with a seeded conjugate Gibbs sampler and a two-state
//! forward-filtering, backward-sampling (FFBS) step. `NaN` observations retain
//! their scheduled time positions while infinities are rejected.
//!
//! Independent chains execute on the active Rayon pool (or its global pool),
//! while indexed collection preserves deterministic chain ordering.

use crate::bayesian_forecast::{BayesianForecastError, ForecastQuantile, InverseGammaPrior};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

const SYMMETRY_TOLERANCE: f64 = 1e-10;
type TrendChainPaths = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

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
    fn validate(&self) -> Result<(), BayesianForecastError> {
        if self.initial_mean.iter().any(|value| !value.is_finite()) {
            return Err(invalid_config(
                "initial mean must contain only finite values",
            ));
        }
        validate_covariance(self.initial_covariance)?;
        validate_prior("level-variance prior", self.level_variance_prior)?;
        validate_prior("slope-variance prior", self.slope_variance_prior)?;
        validate_prior(
            "observation-variance prior",
            self.observation_variance_prior,
        )?;
        if self.num_chains == 0 {
            return Err(invalid_config("number of chains must be positive"));
        }
        if self.num_draws == 0 {
            return Err(invalid_config("number of posterior draws must be positive"));
        }
        if self.thinning == 0 {
            return Err(invalid_config("thinning must be positive"));
        }
        self.num_draws
            .checked_mul(self.thinning)
            .and_then(|saved| self.num_warmup.checked_add(saved))
            .ok_or_else(|| {
                invalid_config("warmup, draws, and thinning imply too many iterations")
            })?;
        Ok(())
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

        let chain_paths: Vec<TrendChainPaths> = self
            .chains
            .par_iter()
            .enumerate()
            .map(|(chain_index, posterior_chain)| {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(chain_seed(seed, chain_index, FORECAST_SEED_DOMAIN));
                let mut chain_levels = Vec::with_capacity(posterior_chain.len());
                let mut chain_slopes = Vec::with_capacity(posterior_chain.len());
                let mut chain_observations = Vec::with_capacity(posterior_chain.len());
                for draw in posterior_chain {
                    validate_positive("level", draw.level_variance)?;
                    validate_positive("slope", draw.slope_variance)?;
                    validate_positive("observation", draw.observation_variance)?;
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
                        level += slope + standard_normal(&mut rng) * level_sd;
                        slope += standard_normal(&mut rng) * slope_sd;
                        let observation = level + standard_normal(&mut rng) * observation_sd;
                        if !level.is_finite() || !slope.is_finite() || !observation.is_finite() {
                            return Err(numerical("posterior predictive simulation overflowed"));
                        }
                        levels.push(level);
                        slopes.push(slope);
                        observations.push(observation);
                    }
                    chain_levels.push(levels);
                    chain_slopes.push(slopes);
                    chain_observations.push(observations);
                }
                Ok((chain_levels, chain_slopes, chain_observations))
            })
            .collect::<Result<_, BayesianForecastError>>()?;

        let mut level_paths = Vec::with_capacity(chain_paths.len());
        let mut slope_paths = Vec::with_capacity(chain_paths.len());
        let mut observation_paths = Vec::with_capacity(chain_paths.len());
        for (chain_levels, chain_slopes, chain_observations) in chain_paths {
            level_paths.push(chain_levels);
            slope_paths.push(chain_slopes);
            observation_paths.push(chain_observations);
        }

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

/// Fit a Bayesian local-linear-trend model with conjugate Gibbs/FFBS.
pub fn fit_bayesian_local_linear_trend(
    observations: &[f64],
    config: &BayesianLocalLinearTrendConfig,
) -> Result<LocalLinearTrendPosterior, BayesianForecastError> {
    config.validate()?;
    validate_observations(observations)?;

    let total_iterations = config.num_warmup + config.num_draws * config.thinning;
    let observed_count = observations.iter().filter(|value| !value.is_nan()).count();
    let transition_count = observations.len() as f64;
    let chains = (0..config.num_chains)
        .into_par_iter()
        .map(|chain_index| {
            let mut rng =
                ChaCha8Rng::seed_from_u64(chain_seed(config.seed, chain_index, FIT_SEED_DOMAIN));
            let mut level_variance = prior_mode(config.level_variance_prior);
            let mut slope_variance = prior_mode(config.slope_variance_prior);
            let mut observation_variance = prior_mode(config.observation_variance_prior);
            let mut posterior_draws = Vec::with_capacity(config.num_draws);

            for iteration in 0..total_iterations {
                // Index zero is x[-1], followed by x[0] through x[T - 1].
                let states = sample_states_ffbs(
                    observations,
                    config.initial_mean,
                    config.initial_covariance,
                    level_variance,
                    slope_variance,
                    observation_variance,
                    &mut rng,
                )?;

                let mut level_sum_sq = 0.0;
                let mut slope_sum_sq = 0.0;
                for pair in states.windows(2) {
                    let level_residual = pair[1][0] - pair[0][0] - pair[0][1];
                    let slope_residual = pair[1][1] - pair[0][1];
                    level_sum_sq += level_residual * level_residual;
                    slope_sum_sq += slope_residual * slope_residual;
                }
                level_variance = sample_inverse_gamma(
                    config.level_variance_prior.shape + transition_count / 2.0,
                    config.level_variance_prior.scale + level_sum_sq / 2.0,
                    &mut rng,
                )?;
                slope_variance = sample_inverse_gamma(
                    config.slope_variance_prior.shape + transition_count / 2.0,
                    config.slope_variance_prior.scale + slope_sum_sq / 2.0,
                    &mut rng,
                )?;

                let observation_sum_sq: f64 = observations
                    .iter()
                    .zip(&states[1..])
                    .filter(|(observation, _)| !observation.is_nan())
                    .map(|(observation, state)| {
                        let residual = observation - state[0];
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
                    let terminal = states.last().expect("observations are non-empty");
                    posterior_draws.push(LocalLinearTrendPosteriorDraw {
                        level_variance,
                        slope_variance,
                        observation_variance,
                        terminal_level: terminal[0],
                        terminal_slope: terminal[1],
                    });
                }
            }
            debug_assert_eq!(posterior_draws.len(), config.num_draws);
            Ok(posterior_draws)
        })
        .collect::<Result<Vec<_>, BayesianForecastError>>()?;
    Ok(LocalLinearTrendPosterior { chains })
}

#[allow(clippy::too_many_arguments)]
fn sample_states_ffbs(
    observations: &[f64],
    initial_mean: [f64; 2],
    initial_covariance: [f64; 4],
    level_variance: f64,
    slope_variance: f64,
    observation_variance: f64,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<[f64; 2]>, BayesianForecastError> {
    validate_positive("level", level_variance)?;
    validate_positive("slope", slope_variance)?;
    validate_positive("observation", observation_variance)?;

    let len = observations.len();
    let mut filtered_means = Vec::with_capacity(len + 1);
    let mut filtered_covariances = Vec::with_capacity(len + 1);
    let mut predicted_means = Vec::with_capacity(len);
    let mut predicted_covariances = Vec::with_capacity(len);
    filtered_means.push(initial_mean);
    filtered_covariances.push(initial_covariance);

    for (time, &observation) in observations.iter().enumerate() {
        let mean = filtered_means[time];
        let covariance = filtered_covariances[time];
        let predicted_mean = [mean[0] + mean[1], mean[1]];
        let predicted_covariance = [
            covariance[0] + covariance[1] + covariance[2] + covariance[3] + level_variance,
            covariance[1] + covariance[3],
            covariance[2] + covariance[3],
            covariance[3] + slope_variance,
        ];
        validate_computed_state("predicted", time, predicted_mean, predicted_covariance)?;

        let (filtered_mean, filtered_covariance) = if observation.is_nan() {
            (predicted_mean, predicted_covariance)
        } else {
            let innovation_variance = predicted_covariance[0] + observation_variance;
            validate_positive("innovation", innovation_variance)?;
            let innovation = observation - predicted_mean[0];
            let gain = [
                predicted_covariance[0] / innovation_variance,
                predicted_covariance[2] / innovation_variance,
            ];
            let filtered_mean = [
                predicted_mean[0] + gain[0] * innovation,
                predicted_mean[1] + gain[1] * innovation,
            ];
            // Algebraically equivalent to the Joseph update for scalar H=[1,0].
            let filtered_covariance = [
                predicted_covariance[0] * observation_variance / innovation_variance,
                predicted_covariance[1] * observation_variance / innovation_variance,
                predicted_covariance[2] * observation_variance / innovation_variance,
                predicted_covariance[3]
                    - predicted_covariance[2] * predicted_covariance[1] / innovation_variance,
            ];
            (filtered_mean, symmetrized(filtered_covariance))
        };
        validate_computed_state("filtered", time, filtered_mean, filtered_covariance)?;
        predicted_means.push(predicted_mean);
        predicted_covariances.push(predicted_covariance);
        filtered_means.push(filtered_mean);
        filtered_covariances.push(filtered_covariance);
    }

    let mut states = vec![[0.0; 2]; len + 1];
    states[len] = sample_bivariate_normal(filtered_means[len], filtered_covariances[len], rng)?;
    for index in (0..len).rev() {
        let covariance = filtered_covariances[index];
        // C F' for F=[[1,1],[0,1]].
        let numerator = [
            covariance[0] + covariance[1],
            covariance[1],
            covariance[2] + covariance[3],
            covariance[3],
        ];
        let prediction_factor = cholesky_2x2(predicted_covariances[index])
            .map_err(|_| numerical("predicted covariance could not be factored"))?;
        let first_gain_row = solve_cholesky_2x2(prediction_factor, [numerator[0], numerator[1]]);
        let second_gain_row = solve_cholesky_2x2(prediction_factor, [numerator[2], numerator[3]]);
        let gain = [
            first_gain_row[0],
            first_gain_row[1],
            second_gain_row[0],
            second_gain_row[1],
        ];
        let delta = [
            states[index + 1][0] - predicted_means[index][0],
            states[index + 1][1] - predicted_means[index][1],
        ];
        let correction = mat_vec_2x2(gain, delta);
        let conditional_mean = [
            filtered_means[index][0] + correction[0],
            filtered_means[index][1] + correction[1],
        ];
        // Joseph form for C - J P J': with A = I - J F and
        // P = F C F' + Q, the conditional covariance is
        // A C A' + J Q J'. This avoids catastrophic cancellation.
        let residual_transition = [
            1.0 - gain[0],
            -(gain[0] + gain[1]),
            -gain[2],
            1.0 - gain[2] - gain[3],
        ];
        let propagated = mat_mul_2x2(
            mat_mul_2x2(residual_transition, covariance),
            transpose_2x2(residual_transition),
        );
        let process_contribution = mat_mul_2x2(
            mat_mul_2x2(gain, [level_variance, 0.0, 0.0, slope_variance]),
            transpose_2x2(gain),
        );
        let conditional_covariance = symmetrized(add_2x2(propagated, process_contribution));
        states[index] = sample_bivariate_normal(conditional_mean, conditional_covariance, rng)?;
    }
    Ok(states)
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
    if observations.iter().filter(|value| !value.is_nan()).count() < 3 {
        return Err(BayesianForecastError::InvalidObservations(
            "at least three finite observations are required for a local linear trend".into(),
        ));
    }
    Ok(())
}

fn validate_prior(name: &str, prior: InverseGammaPrior) -> Result<(), BayesianForecastError> {
    if !prior.shape.is_finite() || prior.shape <= 0.0 {
        return Err(invalid_config(&format!(
            "{name} shape must be finite and strictly positive"
        )));
    }
    if !prior.scale.is_finite() || prior.scale <= 0.0 {
        return Err(invalid_config(&format!(
            "{name} scale must be finite and strictly positive"
        )));
    }
    Ok(())
}

fn validate_covariance(covariance: [f64; 4]) -> Result<(), BayesianForecastError> {
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(invalid_config(
            "initial covariance must contain only finite values",
        ));
    }
    let scale = 1.0_f64.max(covariance[1].abs()).max(covariance[2].abs());
    if (covariance[1] - covariance[2]).abs() > SYMMETRY_TOLERANCE * scale {
        return Err(invalid_config("initial covariance must be symmetric"));
    }
    cholesky_2x2(symmetrized(covariance))
        .map_err(|_| invalid_config("initial covariance must be strictly positive definite"))?;
    Ok(())
}

fn validate_computed_state(
    name: &str,
    time: usize,
    mean: [f64; 2],
    covariance: [f64; 4],
) -> Result<(), BayesianForecastError> {
    if mean
        .iter()
        .chain(covariance.iter())
        .any(|value| !value.is_finite())
    {
        return Err(numerical(&format!(
            "{name} state at time {time} contains a non-finite value"
        )));
    }
    cholesky_2x2(symmetrized(covariance)).map_err(|_| {
        numerical(&format!(
            "{name} covariance at time {time} is not positive definite"
        ))
    })?;
    Ok(())
}

fn sample_inverse_gamma(
    shape: f64,
    scale: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    if !shape.is_finite() || shape <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        return Err(numerical("invalid inverse-gamma posterior parameters"));
    }
    let gamma = Gamma::new(shape, 1.0 / scale)
        .map_err(|error| numerical(&format!("could not construct gamma distribution: {error}")))?;
    let variance = 1.0 / gamma.sample(rng);
    validate_positive("sampled", variance)?;
    Ok(variance)
}

fn sample_bivariate_normal(
    mean: [f64; 2],
    covariance: [f64; 4],
    rng: &mut ChaCha8Rng,
) -> Result<[f64; 2], BayesianForecastError> {
    let factor = cholesky_2x2(symmetrized(covariance))
        .map_err(|_| numerical("normal covariance is not positive definite"))?;
    let z0 = standard_normal(rng);
    let z1 = standard_normal(rng);
    let draw = [
        mean[0] + factor[0] * z0,
        mean[1] + factor[2] * z0 + factor[3] * z1,
    ];
    if draw.iter().any(|value| !value.is_finite()) {
        return Err(numerical("normal simulation produced a non-finite draw"));
    }
    Ok(draw)
}

fn cholesky_2x2(matrix: [f64; 4]) -> Result<[f64; 4], ()> {
    if !matrix[0].is_finite() || matrix[0] <= 0.0 {
        return Err(());
    }
    let l00 = matrix[0].sqrt();
    let l10 = matrix[2] / l00;
    let remainder = matrix[3] - l10 * l10;
    if !remainder.is_finite() || remainder <= 0.0 {
        return Err(());
    }
    Ok([l00, 0.0, l10, remainder.sqrt()])
}

fn solve_cholesky_2x2(factor: [f64; 4], right_hand_side: [f64; 2]) -> [f64; 2] {
    let y0 = right_hand_side[0] / factor[0];
    let y1 = (right_hand_side[1] - factor[2] * y0) / factor[3];
    let x1 = y1 / factor[3];
    let x0 = (y0 - factor[2] * x1) / factor[0];
    [x0, x1]
}

fn mat_mul_2x2(left: [f64; 4], right: [f64; 4]) -> [f64; 4] {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

fn mat_vec_2x2(matrix: [f64; 4], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1],
        matrix[2] * vector[0] + matrix[3] * vector[1],
    ]
}

fn transpose_2x2(matrix: [f64; 4]) -> [f64; 4] {
    [matrix[0], matrix[2], matrix[1], matrix[3]]
}

fn add_2x2(left: [f64; 4], right: [f64; 4]) -> [f64; 4] {
    [
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
    ]
}

fn symmetrized(matrix: [f64; 4]) -> [f64; 4] {
    let off_diagonal = 0.5 * (matrix[1] + matrix[2]);
    [matrix[0], off_diagonal, off_diagonal, matrix[3]]
}

fn prior_mode(prior: InverseGammaPrior) -> f64 {
    prior.scale / (prior.shape + 1.0)
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

fn validate_positive(name: &str, variance: f64) -> Result<(), BayesianForecastError> {
    if !variance.is_finite() || variance <= 0.0 {
        return Err(numerical(&format!(
            "{name} variance must be finite and strictly positive"
        )));
    }
    Ok(())
}

const FIT_SEED_DOMAIN: u64 = 0x4649_545F_5452_454E;
const FORECAST_SEED_DOMAIN: u64 = 0x4652_4353_545F_5452;

fn chain_seed(seed: u64, chain_index: usize, domain: u64) -> u64 {
    let mut value = seed
        .wrapping_add(domain)
        .wrapping_add((chain_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn path_means(paths: &[Vec<Vec<f64>>]) -> Result<Vec<f64>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    let mut means = vec![0.0; horizon];
    let mut count = 0usize;
    for path in paths.iter().flatten() {
        for (mean, value) in means.iter_mut().zip(path) {
            *mean += value;
        }
        count += 1;
    }
    for mean in &mut means {
        *mean /= count as f64;
    }
    Ok(means)
}

fn path_quantiles(
    paths: &[Vec<Vec<f64>>],
    probabilities: &[f64],
) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    for &probability in probabilities {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(invalid_config(
                "quantile probabilities must be finite and between zero and one",
            ));
        }
    }
    let ordered_by_step: Vec<Vec<f64>> = (0..horizon)
        .map(|step| {
            let mut values: Vec<f64> = paths.iter().flatten().map(|path| path[step]).collect();
            values.sort_by(f64::total_cmp);
            values
        })
        .collect();
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
        return Err(invalid_config(
            "forecast must contain at least one non-empty path per chain",
        ));
    }
    if paths.iter().flatten().any(|path| path.len() != horizon) {
        return Err(invalid_config(
            "forecast paths must all have the same horizon",
        ));
    }
    if paths
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(numerical("forecast contains a non-finite value"));
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

fn invalid_config(message: &str) -> BayesianForecastError {
    BayesianForecastError::InvalidConfiguration(message.into())
}

fn numerical(message: &str) -> BayesianForecastError {
    BayesianForecastError::NumericalFailure(message.into())
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

    #[test]
    fn ffbs_midpoint_and_terminal_moments_match_existing_kalman_smoother() {
        use crate::state_space::LinearGaussianStateSpace;

        let observations = [0.2, 0.5, 0.9, f64::NAN, f64::NAN];
        let initial_mean = [-0.1, 0.15];
        let initial_covariance = [1.4, 0.25, 0.25, 0.8];
        let level_variance = 0.3;
        let slope_variance = 0.08;
        let observation_variance = 0.5;
        let smoother = LinearGaussianStateSpace::new(
            2,
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![level_variance, 0.0, 0.0, slope_variance],
            observation_variance,
            initial_mean.to_vec(),
            initial_covariance.to_vec(),
        )
        .unwrap()
        .smooth(&observations)
        .unwrap();
        let midpoint = 2;
        let expected_midpoint_mean = &smoother.smoothed_means[midpoint];
        let expected_midpoint_covariance = &smoother.smoothed_covariances[midpoint];
        let expected_terminal_mean = smoother.smoothed_means.last().unwrap();
        let expected_terminal_covariance = smoother.smoothed_covariances.last().unwrap();

        let draws = 30_000;
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut midpoint_sum = [0.0; 2];
        let mut midpoint_products = [0.0; 4];
        let mut terminal_sum = [0.0; 2];
        let mut terminal_products = [0.0; 4];
        for _ in 0..draws {
            let states = sample_states_ffbs(
                &observations,
                initial_mean,
                initial_covariance,
                level_variance,
                slope_variance,
                observation_variance,
                &mut rng,
            )
            .unwrap();
            for (state, sum, products) in [
                (
                    states[midpoint + 1],
                    &mut midpoint_sum,
                    &mut midpoint_products,
                ),
                (
                    *states.last().unwrap(),
                    &mut terminal_sum,
                    &mut terminal_products,
                ),
            ] {
                sum[0] += state[0];
                sum[1] += state[1];
                products[0] += state[0] * state[0];
                products[1] += state[0] * state[1];
                products[2] += state[1] * state[0];
                products[3] += state[1] * state[1];
            }
        }
        for (sum, products, expected_mean, expected_covariance) in [
            (
                midpoint_sum,
                midpoint_products,
                expected_midpoint_mean,
                expected_midpoint_covariance,
            ),
            (
                terminal_sum,
                terminal_products,
                expected_terminal_mean,
                expected_terminal_covariance,
            ),
        ] {
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
    }

    #[test]
    fn ffbs_smoothing_stays_finite_at_extreme_covariance_scales() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let states = sample_states_ffbs(
            &[f64::NAN; 4],
            [0.0, 0.0],
            [1e200, 0.0, 0.0, 1e200],
            1e-7,
            1e-12,
            1.0,
            &mut rng,
        )
        .unwrap();
        assert!(states.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn recovers_signal_and_produces_widening_forecasts() {
        let true_level_variance: f64 = 0.12;
        let true_slope_variance: f64 = 0.025;
        let true_observation_variance: f64 = 0.35;
        let mut rng = ChaCha8Rng::seed_from_u64(810);
        let mut level = 0.0;
        let mut slope = 0.08;
        let mut observations = Vec::with_capacity(280);
        for _ in 0..280 {
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
        assert!((means[0] / count - true_level_variance).abs() < 0.12);
        assert!((means[1] / count - true_slope_variance).abs() < 0.025);
        assert!((means[2] / count - true_observation_variance).abs() < 0.18);

        let forecast = posterior.forecast(15, 812).unwrap();
        let intervals = forecast.observation_quantiles(&[0.1, 0.9]).unwrap();
        let first_width = intervals[1].values[0] - intervals[0].values[0];
        let last_width = intervals[1].values[14] - intervals[0].values[14];
        assert!(last_width > first_width, "{first_width} vs {last_width}");
    }
}
