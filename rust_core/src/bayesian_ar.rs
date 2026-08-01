//! Conjugate Bayesian autoregression and posterior-predictive forecasting.
//!
//! For a caller-selected order `p >= 1`, the fitted model is
//!
//! ```text
//! y[t] = intercept + phi[1] y[t - 1] + ... + phi[p] y[t - p] + epsilon[t]
//! epsilon[t] ~ Normal(0, innovation_variance)
//! ```
//!
//! The likelihood is conditional on the first `p` observations. The prior is
//! the explicit Normal-Inverse-Gamma distribution
//!
//! ```text
//! innovation_variance ~ InverseGamma(variance_shape, variance_scale)
//! coefficients | innovation_variance
//!     ~ Normal(coefficient_mean, innovation_variance * coefficient_precision^-1)
//! ```
//!
//! where the inverse-gamma density is proportional to
//! `x^(-shape - 1) * exp(-scale / x)`. Coefficients are ordered as
//! `[intercept, lag_1, ..., lag_p]`.
//!
//! This is a direct observed-data autoregression, distinct from a latent
//! state-space AR model with an additional observation-noise layer.
//!
//! Posterior coefficient draws are deliberately **not** restricted to the
//! stationary region. This preserves the exact conjugate posterior. Users
//! who require a stationary model should inspect/filter draws explicitly or
//! use a correctly normalized constrained prior; no silent clipping occurs.
//!
//! Independent chains execute on the active Rayon pool (or its global pool),
//! while indexed collection preserves deterministic chain ordering.

use crate::bayesian_forecast::{BayesianForecastError, ForecastQuantile};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

type ArChainPaths = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Explicit Normal-Inverse-Gamma prior for an AR(p) regression.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalInverseGammaPrior {
    /// Mean of `[intercept, lag_1, ..., lag_p]`.
    pub coefficient_mean: Vec<f64>,
    /// Symmetric positive-definite precision matrix before variance scaling.
    pub coefficient_precision: Vec<Vec<f64>>,
    /// Shape of the innovation-variance inverse-gamma prior.
    pub variance_shape: f64,
    /// Scale of the innovation-variance inverse-gamma prior.
    pub variance_scale: f64,
}

impl NormalInverseGammaPrior {
    /// Construct and validate a prior.
    ///
    /// The coefficient dimension must be at least two (intercept plus one
    /// lag), and `coefficient_precision` must be finite, symmetric, and
    /// positive definite.
    pub fn new(
        coefficient_mean: Vec<f64>,
        coefficient_precision: Vec<Vec<f64>>,
        variance_shape: f64,
        variance_scale: f64,
    ) -> Result<Self, BayesianForecastError> {
        let prior = Self {
            coefficient_mean,
            coefficient_precision,
            variance_shape,
            variance_scale,
        };
        let dimension = prior.coefficient_mean.len();
        if dimension < 2 {
            return Err(invalid_configuration(
                "coefficient prior must include an intercept and at least one lag",
            ));
        }
        dimension.checked_mul(dimension).ok_or_else(|| {
            invalid_configuration("coefficient prior precision dimensions overflow")
        })?;
        validate_prior(&prior, dimension)?;
        cholesky(&prior.coefficient_precision, "coefficient prior precision")?;
        Ok(prior)
    }
}

/// Sampling configuration for a conjugate Bayesian AR(p) model.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianArConfig {
    pub order: usize,
    pub prior: NormalInverseGammaPrior,
    pub num_chains: usize,
    pub num_draws: usize,
    pub seed: u64,
}

/// One joint posterior draw.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianArPosteriorDraw {
    /// `[intercept, lag_1, ..., lag_p]`.
    pub coefficients: Vec<f64>,
    pub innovation_variance: f64,
}

/// Exact conjugate posterior samples, preserving chain/draw provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianArPosterior {
    pub order: usize,
    /// The final `order` observations in chronological order.
    pub terminal_observations: Vec<f64>,
    /// Joint samples indexed as `[chain][draw]`.
    pub chains: Vec<Vec<BayesianArPosteriorDraw>>,
}

impl BayesianArPosterior {
    /// Simulate coherent recursive conditional-mean and observation paths.
    ///
    /// At each horizon the conditional mean uses the already simulated
    /// observations from earlier horizons. Consequently each mean path is
    /// paired with, and conditional on, its corresponding observation path.
    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<BayesianArForecast, BayesianForecastError> {
        if horizon == 0 {
            return Err(invalid_configuration("forecast horizon must be positive"));
        }
        if self.order == 0 {
            return Err(invalid_configuration("AR order must be positive"));
        }
        if self.terminal_observations.len() != self.order
            || self
                .terminal_observations
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(invalid_configuration(
                "posterior terminal observations must contain one finite value per lag",
            ));
        }
        if self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(invalid_configuration(
                "every posterior chain must contain at least one draw",
            ));
        }

        let expected_coefficients = self.order.checked_add(1).ok_or_else(|| {
            invalid_configuration("AR order is too large to represent its coefficients")
        })?;
        self.order
            .checked_add(horizon)
            .ok_or_else(|| invalid_configuration("AR forecast history length overflows"))?;
        for chain in &self.chains {
            chain
                .len()
                .checked_mul(horizon)
                .ok_or_else(|| invalid_configuration("AR forecast path dimensions overflow"))?;
        }
        let chain_paths: Vec<ArChainPaths> = self
            .chains
            .par_iter()
            .enumerate()
            .map(|(chain_index, chain)| {
                let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(
                    seed,
                    chain_index,
                    FORECAST_SEED_DOMAIN,
                ));
                let mut chain_means = Vec::with_capacity(chain.len());
                let mut chain_observations = Vec::with_capacity(chain.len());
                for draw in chain {
                    if draw.coefficients.len() != expected_coefficients
                        || draw.coefficients.iter().any(|value| !value.is_finite())
                    {
                        return Err(invalid_configuration(
                            "each posterior draw must contain finite intercept and lag coefficients",
                        ));
                    }
                    validate_positive_finite("innovation variance", draw.innovation_variance)?;

                    let mut history = self.terminal_observations.clone();
                    history.reserve(horizon);
                    let mut means = Vec::with_capacity(horizon);
                    let mut observations = Vec::with_capacity(horizon);
                    let innovation_sd = draw.innovation_variance.sqrt();
                    for _ in 0..horizon {
                        let mut mean = draw.coefficients[0];
                        for lag in 1..=self.order {
                            mean += draw.coefficients[lag] * history[history.len() - lag];
                        }
                        let observation = mean + standard_normal(&mut rng) * innovation_sd;
                        if !mean.is_finite() || !observation.is_finite() {
                            return Err(BayesianForecastError::NumericalFailure(
                                "recursive AR forecast overflowed; posterior draws are unconstrained and may be explosive"
                                    .into(),
                            ));
                        }
                        means.push(mean);
                        observations.push(observation);
                        history.push(observation);
                    }
                    chain_means.push(means);
                    chain_observations.push(observations);
                }
                Ok((chain_means, chain_observations))
            })
            .collect::<Result<_, BayesianForecastError>>()?;

        let mut conditional_mean_paths = Vec::with_capacity(chain_paths.len());
        let mut observation_paths = Vec::with_capacity(chain_paths.len());
        for (chain_means, chain_observations) in chain_paths {
            conditional_mean_paths.push(chain_means);
            observation_paths.push(chain_observations);
        }

        Ok(BayesianArForecast {
            conditional_mean_paths,
            observation_paths,
        })
    }
}

/// Posterior-predictive AR(p) paths indexed as `[chain][draw][forecast_step]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianArForecast {
    pub conditional_mean_paths: Vec<Vec<Vec<f64>>>,
    pub observation_paths: Vec<Vec<Vec<f64>>>,
}

impl BayesianArForecast {
    pub fn horizon(&self) -> usize {
        self.observation_paths
            .first()
            .and_then(|chain| chain.first())
            .map_or(0, Vec::len)
    }

    pub fn conditional_mean_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.conditional_mean_paths)
    }

    pub fn observation_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.observation_paths)
    }

    pub fn conditional_mean_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.conditional_mean_paths, probabilities)
    }

    pub fn observation_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.observation_paths, probabilities)
    }
}

/// Fit an exact conjugate Bayesian AR(p) model.
pub fn fit_bayesian_ar(
    observations: &[f64],
    config: &BayesianArConfig,
) -> Result<BayesianArPosterior, BayesianForecastError> {
    let dimension = validate_inputs(observations, config)?;
    let regression_rows = observations.len() - config.order;
    let prior_cholesky = cholesky(&config.prior.coefficient_precision, "prior precision")?;

    let mut posterior_precision = config.prior.coefficient_precision.clone();
    let mut posterior_rhs = matrix_vector_product(
        &config.prior.coefficient_precision,
        &config.prior.coefficient_mean,
    )?;

    for time in config.order..observations.len() {
        let row = regression_row(observations, time, config.order);
        let target = observations[time];
        for column in 0..dimension {
            posterior_rhs[column] += row[column] * target;
            for other in 0..dimension {
                posterior_precision[column][other] += row[column] * row[other];
            }
        }
    }
    ensure_finite_vector("posterior right-hand side", &posterior_rhs)?;
    ensure_finite_matrix("posterior precision", &posterior_precision)?;

    let posterior_cholesky = cholesky(&posterior_precision, "posterior precision")?;
    let posterior_mean = solve_cholesky(&posterior_cholesky, &posterior_rhs)?;

    // Stable completion-of-squares form for the posterior scale.
    let mut residual_sum_sq = 0.0;
    for time in config.order..observations.len() {
        let row = regression_row(observations, time, config.order);
        let residual = observations[time] - dot(&row, &posterior_mean);
        residual_sum_sq += residual * residual;
    }
    let mean_delta: Vec<f64> = posterior_mean
        .iter()
        .zip(&config.prior.coefficient_mean)
        .map(|(posterior, prior)| posterior - prior)
        .collect();
    let prior_penalty = quadratic_form(&config.prior.coefficient_precision, &mean_delta)?;
    let posterior_shape = config.prior.variance_shape + regression_rows as f64 / 2.0;
    let posterior_scale = config.prior.variance_scale + 0.5 * (residual_sum_sq + prior_penalty);
    validate_positive_finite("posterior variance shape", posterior_shape)?;
    validate_positive_finite("posterior variance scale", posterior_scale)?;

    let chains = (0..config.num_chains)
        .into_par_iter()
        .map(|chain_index| {
            let mut rng =
                ChaCha8Rng::seed_from_u64(chain_seed(config.seed, chain_index, FIT_SEED_DOMAIN));
            let mut chain = Vec::with_capacity(config.num_draws);
            for _ in 0..config.num_draws {
                let innovation_variance =
                    sample_inverse_gamma(posterior_shape, posterior_scale, &mut rng)?;
                let mut standard_draw = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    standard_draw.push(standard_normal(&mut rng));
                }
                let precision_scaled_draw =
                    solve_transposed_lower(&posterior_cholesky, &standard_draw)?;
                let innovation_sd = innovation_variance.sqrt();
                let coefficients: Vec<f64> = posterior_mean
                    .iter()
                    .zip(precision_scaled_draw)
                    .map(|(mean, draw)| mean + innovation_sd * draw)
                    .collect();
                ensure_finite_vector("sampled coefficients", &coefficients)?;
                chain.push(BayesianArPosteriorDraw {
                    coefficients,
                    innovation_variance,
                });
            }
            Ok(chain)
        })
        .collect::<Result<Vec<_>, BayesianForecastError>>()?;

    // Keep the prior factor alive until all validation and posterior arithmetic
    // are complete; computing it above is the SPD validation for the prior.
    drop(prior_cholesky);
    Ok(BayesianArPosterior {
        order: config.order,
        terminal_observations: observations[observations.len() - config.order..].to_vec(),
        chains,
    })
}

fn validate_inputs(
    observations: &[f64],
    config: &BayesianArConfig,
) -> Result<usize, BayesianForecastError> {
    if config.order == 0 {
        return Err(invalid_configuration("AR order must be at least one"));
    }
    let dimension = config.order.checked_add(1).ok_or_else(|| {
        invalid_configuration("AR order is too large to represent its coefficients")
    })?;
    dimension
        .checked_mul(dimension)
        .ok_or_else(|| invalid_configuration("AR coefficient precision dimensions overflow"))?;
    if observations.len() <= config.order {
        return Err(BayesianForecastError::InvalidObservations(format!(
            "AR({}) requires more than {} observations for its conditional likelihood",
            config.order, config.order
        )));
    }
    if observations.iter().any(|value| !value.is_finite()) {
        return Err(BayesianForecastError::InvalidObservations(
            "AR observations must all be finite; missing values are not currently supported".into(),
        ));
    }
    if config.num_chains == 0 {
        return Err(invalid_configuration("number of chains must be positive"));
    }
    if config.num_draws == 0 {
        return Err(invalid_configuration(
            "number of posterior draws must be positive",
        ));
    }
    config
        .num_chains
        .checked_mul(config.num_draws)
        .ok_or_else(|| invalid_configuration("chain and draw counts overflow"))?;
    validate_prior(&config.prior, dimension)?;
    Ok(dimension)
}

fn validate_prior(
    prior: &NormalInverseGammaPrior,
    dimension: usize,
) -> Result<(), BayesianForecastError> {
    if prior.coefficient_mean.len() != dimension {
        return Err(invalid_configuration(format!(
            "coefficient prior mean must have length {dimension}"
        )));
    }
    if prior
        .coefficient_mean
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid_configuration(
            "coefficient prior mean must contain only finite values",
        ));
    }
    if prior.coefficient_precision.len() != dimension
        || prior
            .coefficient_precision
            .iter()
            .any(|row| row.len() != dimension)
    {
        return Err(invalid_configuration(format!(
            "coefficient prior precision must have shape {dimension} by {dimension}"
        )));
    }
    if prior
        .coefficient_precision
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(invalid_configuration(
            "coefficient prior precision must contain only finite values",
        ));
    }
    validate_symmetric(&prior.coefficient_precision)?;
    validate_positive_finite("variance prior shape", prior.variance_shape)?;
    validate_positive_finite("variance prior scale", prior.variance_scale)?;
    Ok(())
}

fn regression_row(observations: &[f64], time: usize, order: usize) -> Vec<f64> {
    let mut row = Vec::with_capacity(order + 1);
    row.push(1.0);
    for lag in 1..=order {
        row.push(observations[time - lag]);
    }
    row
}

fn validate_symmetric(matrix: &[Vec<f64>]) -> Result<(), BayesianForecastError> {
    for (row_index, row) in matrix.iter().enumerate() {
        for (column_index, &value) in row.iter().take(row_index).enumerate() {
            let transposed = matrix[column_index][row_index];
            let scale = value.abs().max(transposed.abs()).max(1.0);
            if (value - transposed).abs() > 1e-12 * scale {
                return Err(invalid_configuration(
                    "coefficient prior precision must be symmetric",
                ));
            }
        }
    }
    Ok(())
}

fn cholesky(matrix: &[Vec<f64>], name: &str) -> Result<Vec<Vec<f64>>, BayesianForecastError> {
    let dimension = matrix.len();
    let mut lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let correction = lower[row][..column]
                .iter()
                .zip(&lower[column][..column])
                .map(|(row_value, column_value)| row_value * column_value)
                .sum::<f64>();
            let value = matrix[row][column] - correction;
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(invalid_configuration(format!(
                        "{name} must be positive definite"
                    )));
                }
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
                if !lower[row][column].is_finite() {
                    return Err(BayesianForecastError::NumericalFailure(format!(
                        "{name} factorization produced a non-finite value"
                    )));
                }
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky(
    lower: &[Vec<f64>],
    right_hand_side: &[f64],
) -> Result<Vec<f64>, BayesianForecastError> {
    let dimension = lower.len();
    let mut intermediate = vec![0.0; dimension];
    for row in 0..dimension {
        let mut value = right_hand_side[row];
        for (coefficient, solved) in lower[row][..row].iter().zip(&intermediate[..row]) {
            value -= coefficient * solved;
        }
        intermediate[row] = value / lower[row][row];
    }
    solve_transposed_lower(lower, &intermediate)
}

fn solve_transposed_lower(
    lower: &[Vec<f64>],
    right_hand_side: &[f64],
) -> Result<Vec<f64>, BayesianForecastError> {
    let dimension = lower.len();
    let mut result = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = right_hand_side[row];
        for column in row + 1..dimension {
            value -= lower[column][row] * result[column];
        }
        result[row] = value / lower[row][row];
    }
    ensure_finite_vector("linear solve", &result)?;
    Ok(result)
}

fn matrix_vector_product(
    matrix: &[Vec<f64>],
    vector: &[f64],
) -> Result<Vec<f64>, BayesianForecastError> {
    let result: Vec<f64> = matrix.iter().map(|row| dot(row, vector)).collect();
    ensure_finite_vector("matrix-vector product", &result)?;
    Ok(result)
}

fn quadratic_form(matrix: &[Vec<f64>], vector: &[f64]) -> Result<f64, BayesianForecastError> {
    let product = matrix_vector_product(matrix, vector)?;
    let value = dot(vector, &product);
    if !value.is_finite() || value < -1e-10 {
        return Err(BayesianForecastError::NumericalFailure(
            "prior quadratic form is invalid".into(),
        ));
    }
    Ok(value.max(0.0))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn sample_inverse_gamma(
    shape: f64,
    scale: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    let gamma = Gamma::new(shape, 1.0 / scale).map_err(|error| {
        BayesianForecastError::NumericalFailure(format!(
            "could not construct gamma distribution: {error}"
        ))
    })?;
    let variance = 1.0 / gamma.sample(rng);
    validate_positive_finite("sampled innovation variance", variance)?;
    Ok(variance)
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

const FIT_SEED_DOMAIN: u64 = 0x4649_545F_4152_5F50;
const FORECAST_SEED_DOMAIN: u64 = 0x4652_4353_545F_4152;

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
    let path_count = paths.iter().map(Vec::len).sum::<usize>();
    let mut means = vec![0.0; horizon];
    for path in paths.iter().flatten() {
        for (mean, value) in means.iter_mut().zip(path) {
            *mean += value;
        }
    }
    for mean in &mut means {
        *mean /= path_count as f64;
    }
    ensure_finite_vector("forecast means", &means)?;
    Ok(means)
}

fn path_quantiles(
    paths: &[Vec<Vec<f64>>],
    probabilities: &[f64],
) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
    let horizon = validate_paths(paths)?;
    for probability in probabilities {
        if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
            return Err(invalid_configuration(
                "quantile probabilities must be finite and between zero and one",
            ));
        }
    }
    let mut by_step = Vec::with_capacity(horizon);
    for step in 0..horizon {
        let mut values: Vec<f64> = paths.iter().flatten().map(|path| path[step]).collect();
        values.sort_by(f64::total_cmp);
        by_step.push(values);
    }
    Ok(probabilities
        .iter()
        .map(|probability| ForecastQuantile {
            probability: *probability,
            values: by_step
                .iter()
                .map(|ordered| interpolated_quantile(ordered, *probability))
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
        return Err(invalid_configuration(
            "forecast must contain at least one non-empty path per chain",
        ));
    }
    if paths.iter().flatten().any(|path| path.len() != horizon) {
        return Err(invalid_configuration(
            "forecast paths must all have the same horizon",
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

fn ensure_finite_vector(name: &str, values: &[f64]) -> Result<(), BayesianForecastError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BayesianForecastError::NumericalFailure(format!(
            "{name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn ensure_finite_matrix(name: &str, matrix: &[Vec<f64>]) -> Result<(), BayesianForecastError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(BayesianForecastError::NumericalFailure(format!(
            "{name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_positive_finite(name: &str, value: f64) -> Result<(), BayesianForecastError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid_configuration(format!(
            "{name} must be finite and strictly positive"
        )));
    }
    Ok(())
}

fn invalid_configuration(message: impl Into<String>) -> BayesianForecastError {
    BayesianForecastError::InvalidConfiguration(message.into())
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

    #[test]
    fn fit_and_forecast_are_bitwise_identical_across_rayon_pool_sizes() {
        let observations = simulate_ar2(120, 43);
        let fit_config = config(2, 44);
        let run = || {
            let posterior = fit_bayesian_ar(&observations, &fit_config).unwrap();
            let forecast = posterior.forecast(6, 45).unwrap();
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

    fn diagonal_prior(order: usize) -> NormalInverseGammaPrior {
        let dimension = order + 1;
        let mut precision = vec![vec![0.0; dimension]; dimension];
        for (index, row) in precision.iter_mut().enumerate() {
            row[index] = 0.2;
        }
        NormalInverseGammaPrior {
            coefficient_mean: vec![0.0; dimension],
            coefficient_precision: precision,
            variance_shape: 2.5,
            variance_scale: 0.5,
        }
    }

    fn config(order: usize, seed: u64) -> BayesianArConfig {
        BayesianArConfig {
            order,
            prior: diagonal_prior(order),
            num_chains: 2,
            num_draws: 500,
            seed,
        }
    }

    fn simulate_ar2(length: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut values = vec![0.1, -0.1];
        while values.len() < length {
            let end = values.len();
            let mean = 0.35 + 0.65 * values[end - 1] - 0.2 * values[end - 2];
            values.push(mean + 0.4 * standard_normal(&mut rng));
        }
        values
    }

    #[test]
    fn fits_caller_selected_order_with_seeded_chain_draw_shapes() {
        let observations = simulate_ar2(500, 10);
        let posterior = fit_bayesian_ar(&observations, &config(2, 11)).unwrap();
        assert_eq!(posterior.order, 2);
        assert_eq!(posterior.terminal_observations, observations[498..]);
        assert_eq!(posterior.chains.len(), 2);
        assert!(posterior.chains.iter().all(|chain| chain.len() == 500));
        assert!(posterior
            .chains
            .iter()
            .flatten()
            .all(|draw| draw.coefficients.len() == 3));
        assert_eq!(
            posterior,
            fit_bayesian_ar(&observations, &config(2, 11)).unwrap()
        );
        assert_ne!(posterior.chains[0], posterior.chains[1]);

        let draw_count = posterior.chains.iter().map(Vec::len).sum::<usize>() as f64;
        let coefficient_means: Vec<f64> = (0..3)
            .map(|index| {
                posterior
                    .chains
                    .iter()
                    .flatten()
                    .map(|draw| draw.coefficients[index])
                    .sum::<f64>()
                    / draw_count
            })
            .collect();
        assert!((coefficient_means[0] - 0.35).abs() < 0.1);
        assert!((coefficient_means[1] - 0.65).abs() < 0.1);
        assert!((coefficient_means[2] + 0.2).abs() < 0.1);
    }

    #[test]
    fn recursive_forecasts_are_coherent_and_empirical_summaries_are_exact() {
        let posterior = fit_bayesian_ar(&simulate_ar2(200, 20), &config(2, 21)).unwrap();
        let forecast = posterior.forecast(5, 22).unwrap();
        assert_eq!(forecast, posterior.forecast(5, 22).unwrap());
        assert_eq!(forecast.horizon(), 5);
        assert_eq!(forecast.conditional_mean_paths.len(), 2);
        assert!(forecast
            .observation_paths
            .iter()
            .all(|chain| chain.len() == 500));

        for (chain_index, chain) in posterior.chains.iter().enumerate() {
            for (draw_index, draw) in chain.iter().enumerate() {
                let mut history = posterior.terminal_observations.clone();
                for step in 0..5 {
                    let expected = draw.coefficients[0]
                        + draw.coefficients[1] * history[history.len() - 1]
                        + draw.coefficients[2] * history[history.len() - 2];
                    assert_eq!(
                        forecast.conditional_mean_paths[chain_index][draw_index][step],
                        expected
                    );
                    history.push(forecast.observation_paths[chain_index][draw_index][step]);
                }
            }
        }

        let means = forecast.observation_means().unwrap();
        let manual_first = forecast
            .observation_paths
            .iter()
            .flatten()
            .map(|path| path[0])
            .sum::<f64>()
            / 1000.0;
        assert_eq!(means[0], manual_first);
        let quantiles = forecast
            .observation_quantiles(&[0.025, 0.5, 0.975])
            .unwrap();
        let mut ordered_first: Vec<f64> = forecast
            .observation_paths
            .iter()
            .flatten()
            .map(|path| path[0])
            .collect();
        ordered_first.sort_by(f64::total_cmp);
        let quantile_index = 0.025 * (ordered_first.len() - 1) as f64;
        let lower_index = quantile_index.floor() as usize;
        let upper_index = quantile_index.ceil() as usize;
        let weight = quantile_index - lower_index as f64;
        let expected_lower =
            ordered_first[lower_index] * (1.0 - weight) + ordered_first[upper_index] * weight;
        assert_eq!(quantiles[0].values[0], expected_lower);
        for step in 0..5 {
            assert!(quantiles[0].values[step] <= quantiles[1].values[step]);
            assert!(quantiles[1].values[step] <= quantiles[2].values[step]);
        }
    }

    #[test]
    fn posterior_draws_match_conjugate_variance_and_coefficient_scale() {
        let observations = simulate_ar2(400, 30);
        let mut sampling_config = config(2, 31);
        sampling_config.num_chains = 4;
        sampling_config.num_draws = 2_000;
        let posterior = fit_bayesian_ar(&observations, &sampling_config).unwrap();

        let draws: Vec<&BayesianArPosteriorDraw> = posterior.chains.iter().flatten().collect();
        let variance_mean = draws
            .iter()
            .map(|draw| draw.innovation_variance)
            .sum::<f64>()
            / draws.len() as f64;
        assert!((variance_mean - 0.16).abs() < 0.035, "{variance_mean}");

        let lag_one_mean =
            draws.iter().map(|draw| draw.coefficients[1]).sum::<f64>() / draws.len() as f64;
        assert!((lag_one_mean - 0.65).abs() < 0.12, "{lag_one_mean}");
    }

    #[test]
    fn rejects_invalid_orders_data_counts_priors_and_sampling_counts() {
        let observations = [1.0, 2.0, 3.0];
        assert!(NormalInverseGammaPrior::new(
            vec![0.0, 0.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            2.0,
            1.0,
        )
        .is_ok());
        assert!(NormalInverseGammaPrior::new(vec![0.0], vec![vec![1.0]], 2.0, 1.0).is_err());
        let mut invalid = config(1, 1);
        invalid.order = 0;
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
        assert!(fit_bayesian_ar(&[1.0], &config(1, 1)).is_err());
        assert!(fit_bayesian_ar(&[1.0, f64::NAN], &config(1, 1)).is_err());
        assert!(fit_bayesian_ar(&[1.0, f64::INFINITY], &config(1, 1)).is_err());

        let mut invalid = config(1, 1);
        invalid.prior.coefficient_mean.pop();
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
        let mut invalid = config(1, 1);
        invalid.prior.coefficient_precision = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
        let mut invalid = config(1, 1);
        invalid.prior.coefficient_precision[0][1] = 0.1;
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
        let mut invalid = config(1, 1);
        invalid.prior.variance_scale = 0.0;
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
        let mut invalid = config(1, 1);
        invalid.num_chains = 0;
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
        let mut invalid = config(1, 1);
        invalid.num_draws = 0;
        assert!(fit_bayesian_ar(&observations, &invalid).is_err());
    }

    #[test]
    fn unconstrained_explosive_draws_are_not_silently_changed() {
        let posterior = BayesianArPosterior {
            order: 1,
            terminal_observations: vec![2.0],
            chains: vec![vec![BayesianArPosteriorDraw {
                coefficients: vec![0.0, 1.5],
                innovation_variance: 1e-30,
            }]],
        };
        let forecast = posterior.forecast(3, 40).unwrap();
        let means = &forecast.conditional_mean_paths[0][0];
        assert!(means[0] > 2.9 && means[1] > means[0] && means[2] > means[1]);
    }
}
