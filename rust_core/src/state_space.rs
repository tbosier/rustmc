//! Linear Gaussian state-space filtering, smoothing, and forecasting.
//!
//! The observation is scalar while the latent state may have any positive
//! dimension.  Matrices are stored in row-major order and validated when the
//! model is constructed.
//!
//! The supplied initial mean and covariance describe the state immediately
//! before the first observation (`x[-1]`). Filtering first applies the
//! transition and process covariance to obtain the prediction for `x[0]`.

use std::error::Error;
use std::fmt;

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

const SYMMETRY_TOLERANCE: f64 = 1e-10;
const LOG_2_PI: f64 = 1.8378770664093453;

#[derive(Debug, Clone, PartialEq)]
pub enum StateSpaceError {
    InvalidDimension(String),
    InvalidParameter(String),
    NonFinite(String),
    NotSymmetric(String),
    NotPositiveSemidefinite(String),
    NotPositiveDefinite(String),
    InvalidVariance(String),
    NumericalFailure(String),
}

impl fmt::Display for StateSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(message) => write!(f, "invalid dimension: {message}"),
            Self::InvalidParameter(message) => write!(f, "invalid parameter: {message}"),
            Self::NonFinite(message) => write!(f, "non-finite value: {message}"),
            Self::NotSymmetric(message) => write!(f, "matrix is not symmetric: {message}"),
            Self::NotPositiveSemidefinite(message) => {
                write!(f, "matrix is not positive semidefinite: {message}")
            }
            Self::NotPositiveDefinite(message) => {
                write!(f, "matrix is not positive definite: {message}")
            }
            Self::InvalidVariance(message) => write!(f, "invalid variance: {message}"),
            Self::NumericalFailure(message) => write!(f, "numerical failure: {message}"),
        }
    }
}

impl Error for StateSpaceError {}

#[derive(Debug, Clone)]
pub struct KalmanFilterResult {
    pub log_likelihood: f64,
    pub predicted_means: Vec<Vec<f64>>,
    pub predicted_covariances: Vec<Vec<f64>>,
    pub filtered_means: Vec<Vec<f64>>,
    pub filtered_covariances: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct KalmanSmootherResult {
    pub filter: KalmanFilterResult,
    pub smoothed_means: Vec<Vec<f64>>,
    pub smoothed_covariances: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct ForecastResult {
    pub state_means: Vec<Vec<f64>>,
    pub state_covariances: Vec<Vec<f64>>,
    pub observation_means: Vec<f64>,
    pub observation_variances: Vec<f64>,
    /// Joint covariance of future observations, stored row-major as
    /// `steps * steps` entries.
    pub observation_covariance: Vec<f64>,
    /// Prefix-sum means: entry `h - 1` is the mean of observations 1 through h.
    pub cumulative_observation_means: Vec<f64>,
    /// Prefix-sum variances, including all cross-horizon covariance terms.
    pub cumulative_observation_variances: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearGaussianStateSpace {
    dimension: usize,
    transition: Vec<f64>,
    observation: Vec<f64>,
    process_covariance: Vec<f64>,
    observation_variance: f64,
    initial_mean: Vec<f64>,
    initial_covariance: Vec<f64>,
}

impl LinearGaussianStateSpace {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dimension: usize,
        transition: Vec<f64>,
        observation: Vec<f64>,
        process_covariance: Vec<f64>,
        observation_variance: f64,
        initial_mean: Vec<f64>,
        initial_covariance: Vec<f64>,
    ) -> Result<Self, StateSpaceError> {
        if dimension == 0 {
            return Err(StateSpaceError::InvalidDimension(
                "latent dimension must be positive".into(),
            ));
        }
        let square = dimension.checked_mul(dimension).ok_or_else(|| {
            StateSpaceError::InvalidDimension("latent dimension is too large".into())
        })?;
        check_len("transition", transition.len(), square)?;
        check_len("observation", observation.len(), dimension)?;
        check_len("process covariance", process_covariance.len(), square)?;
        check_len("initial mean", initial_mean.len(), dimension)?;
        check_len("initial covariance", initial_covariance.len(), square)?;
        check_finite("transition", &transition)?;
        check_finite("observation", &observation)?;
        check_finite("process covariance", &process_covariance)?;
        check_finite("initial mean", &initial_mean)?;
        check_finite("initial covariance", &initial_covariance)?;
        if !observation_variance.is_finite() || observation_variance <= 0.0 {
            return Err(StateSpaceError::InvalidVariance(
                "observation variance must be finite and strictly positive".into(),
            ));
        }
        check_symmetric("process covariance", &process_covariance, dimension)?;
        check_symmetric("initial covariance", &initial_covariance, dimension)?;
        positive_semidefinite_factor(&process_covariance, dimension).map_err(|_| {
            StateSpaceError::NotPositiveSemidefinite(
                "process covariance must be positive semidefinite".into(),
            )
        })?;
        cholesky(&initial_covariance, dimension).map_err(|_| {
            StateSpaceError::NotPositiveDefinite(
                "initial covariance must be strictly positive definite".into(),
            )
        })?;

        Ok(Self {
            dimension,
            transition,
            observation,
            process_covariance,
            observation_variance,
            initial_mean,
            initial_covariance,
        })
    }

    pub fn local_level(
        process_variance: f64,
        observation_variance: f64,
        initial_mean: f64,
        initial_variance: f64,
    ) -> Result<Self, StateSpaceError> {
        Self::new(
            1,
            vec![1.0],
            vec![1.0],
            vec![process_variance],
            observation_variance,
            vec![initial_mean],
            vec![initial_variance],
        )
    }

    pub fn local_linear_trend(
        level_variance: f64,
        trend_variance: f64,
        observation_variance: f64,
        initial_level: f64,
        initial_trend: f64,
        initial_level_variance: f64,
        initial_trend_variance: f64,
    ) -> Result<Self, StateSpaceError> {
        Self::new(
            2,
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![level_variance, 0.0, 0.0, trend_variance],
            observation_variance,
            vec![initial_level, initial_trend],
            vec![initial_level_variance, 0.0, 0.0, initial_trend_variance],
        )
    }

    /// Construct a local-level model with sum-to-zero dummy seasonality.
    ///
    /// `initial_seasonal_effects` contains one complete cycle in forecast
    /// order and must have `period` finite entries summing to zero. The latent
    /// state contains the level and `period - 1` seasonal states. Deterministic
    /// shift states make the process covariance positive semidefinite rather
    /// than strictly positive definite.
    #[allow(clippy::too_many_arguments)]
    pub fn seasonal_local_level(
        period: usize,
        level_variance: f64,
        seasonal_variance: f64,
        observation_variance: f64,
        initial_level: f64,
        initial_seasonal_effects: Vec<f64>,
        initial_level_variance: f64,
        initial_seasonal_variance: f64,
    ) -> Result<Self, StateSpaceError> {
        if period < 2 {
            return Err(StateSpaceError::InvalidParameter(
                "seasonal period must be at least 2".into(),
            ));
        }
        check_len(
            "initial seasonal effects",
            initial_seasonal_effects.len(),
            period,
        )?;
        check_finite("initial seasonal effects", &initial_seasonal_effects)?;
        let seasonal_scale = initial_seasonal_effects
            .iter()
            .map(|effect| effect.abs())
            .sum::<f64>()
            .max(1.0);
        let seasonal_sum = initial_seasonal_effects.iter().sum::<f64>();
        if seasonal_sum.abs() > SYMMETRY_TOLERANCE * seasonal_scale {
            return Err(StateSpaceError::InvalidParameter(format!(
                "initial seasonal effects must sum to zero; sum is {seasonal_sum}"
            )));
        }
        for (name, variance) in [
            ("level variance", level_variance),
            ("seasonal variance", seasonal_variance),
        ] {
            if !variance.is_finite() || variance < 0.0 {
                return Err(StateSpaceError::InvalidVariance(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        for (name, variance) in [
            ("initial level variance", initial_level_variance),
            ("initial seasonal variance", initial_seasonal_variance),
        ] {
            if !variance.is_finite() || variance <= 0.0 {
                return Err(StateSpaceError::InvalidVariance(format!(
                    "{name} must be finite and strictly positive"
                )));
            }
        }

        let dimension = period;
        let square = dimension.checked_mul(dimension).ok_or_else(|| {
            StateSpaceError::InvalidDimension("seasonal period is too large".into())
        })?;
        let mut transition = vec![0.0; square];
        transition[0] = 1.0;
        for column in 1..dimension {
            transition[dimension + column] = -1.0;
        }
        for row in 2..dimension {
            transition[row * dimension + row - 1] = 1.0;
        }

        let mut observation = vec![0.0; dimension];
        observation[0] = 1.0;
        observation[1] = 1.0;
        let mut process_covariance = vec![0.0; square];
        process_covariance[0] = level_variance;
        process_covariance[dimension + 1] = seasonal_variance;

        let mut initial_mean = vec![0.0; dimension];
        initial_mean[0] = initial_level;
        for state_index in 1..dimension {
            initial_mean[state_index] = initial_seasonal_effects[period - state_index];
        }
        let mut initial_covariance = vec![0.0; square];
        initial_covariance[0] = initial_level_variance;
        for index in 1..dimension {
            initial_covariance[index * dimension + index] = initial_seasonal_variance;
        }

        Self::new(
            dimension,
            transition,
            observation,
            process_covariance,
            observation_variance,
            initial_mean,
            initial_covariance,
        )
    }

    /// Construct a zero-mean stationary AR(1) model with noisy observations.
    ///
    /// The latent state follows `x[t] = coefficient * x[t - 1] + noise[t]`,
    /// where the state noise has variance `process_variance`. Observations are
    /// `y[t] = x[t] + error[t]`, with error variance
    /// `observation_variance`. The initial state is drawn from the stationary
    /// distribution, whose variance is `process_variance / (1 - coefficient^2)`.
    pub fn stationary_ar1(
        coefficient: f64,
        process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self, StateSpaceError> {
        if !coefficient.is_finite() {
            return Err(StateSpaceError::NonFinite(
                "AR(1) coefficient must be finite".into(),
            ));
        }
        if coefficient.abs() >= 1.0 {
            return Err(StateSpaceError::InvalidParameter(
                "AR(1) coefficient must be strictly between -1 and 1 for stationarity".into(),
            ));
        }
        if !process_variance.is_finite() || process_variance <= 0.0 {
            return Err(StateSpaceError::InvalidVariance(
                "process variance must be finite and strictly positive".into(),
            ));
        }
        if !observation_variance.is_finite() || observation_variance <= 0.0 {
            return Err(StateSpaceError::InvalidVariance(
                "observation variance must be finite and strictly positive".into(),
            ));
        }

        let stationary_variance = process_variance / (1.0 - coefficient * coefficient);
        if !stationary_variance.is_finite() || stationary_variance <= 0.0 {
            return Err(StateSpaceError::InvalidVariance(
                "stationary state variance must be finite and strictly positive".into(),
            ));
        }

        Self::new(
            1,
            vec![coefficient],
            vec![1.0],
            vec![process_variance],
            observation_variance,
            vec![0.0],
            vec![stationary_variance],
        )
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn filter(&self, observations: &[f64]) -> Result<KalmanFilterResult, StateSpaceError> {
        validate_observations(observations)?;
        let d = self.dimension;
        let mut previous_mean = self.initial_mean.clone();
        let mut previous_covariance = self.initial_covariance.clone();
        let mut predicted_means = Vec::with_capacity(observations.len());
        let mut predicted_covariances = Vec::with_capacity(observations.len());
        let mut filtered_means = Vec::with_capacity(observations.len());
        let mut filtered_covariances = Vec::with_capacity(observations.len());
        let mut log_likelihood = 0.0;

        for (time, &value) in observations.iter().enumerate() {
            let predicted_mean = mat_vec(&self.transition, &previous_mean, d);
            let mut predicted_covariance = mat_mul_transpose_right(
                &mat_mul(&self.transition, &previous_covariance, d),
                &self.transition,
                d,
            );
            add_assign(&mut predicted_covariance, &self.process_covariance);
            symmetrize(&mut predicted_covariance, d);
            check_computed(
                "predicted state",
                &predicted_mean,
                &predicted_covariance,
                time,
            )?;

            let (filtered_mean, filtered_covariance) = if value.is_nan() {
                (predicted_mean.clone(), predicted_covariance.clone())
            } else {
                let ph = mat_vec(&predicted_covariance, &self.observation, d);
                let innovation_variance = dot(&self.observation, &ph) + self.observation_variance;
                if !innovation_variance.is_finite() || innovation_variance <= 0.0 {
                    return Err(StateSpaceError::NumericalFailure(format!(
                        "innovation variance at time {time} is not finite and positive"
                    )));
                }
                let predicted_observation = dot(&self.observation, &predicted_mean);
                let innovation = value - predicted_observation;
                if !innovation.is_finite() {
                    return Err(StateSpaceError::NumericalFailure(format!(
                        "innovation at time {time} is not finite"
                    )));
                }
                let gain: Vec<f64> = ph.iter().map(|entry| entry / innovation_variance).collect();
                let mut mean = predicted_mean.clone();
                for i in 0..d {
                    mean[i] += gain[i] * innovation;
                }

                // Joseph form is more resistant to roundoff than P - K H P.
                let mut update = identity(d);
                for i in 0..d {
                    for j in 0..d {
                        update[i * d + j] -= gain[i] * self.observation[j];
                    }
                }
                let left = mat_mul(&update, &predicted_covariance, d);
                let mut covariance = mat_mul_transpose_right(&left, &update, d);
                for i in 0..d {
                    for j in 0..d {
                        covariance[i * d + j] += gain[i] * self.observation_variance * gain[j];
                    }
                }
                symmetrize(&mut covariance, d);
                check_computed("filtered state", &mean, &covariance, time)?;

                let contribution = -0.5
                    * (LOG_2_PI
                        + innovation_variance.ln()
                        + innovation * innovation / innovation_variance);
                if !contribution.is_finite() {
                    return Err(StateSpaceError::NumericalFailure(format!(
                        "log-likelihood contribution at time {time} is not finite"
                    )));
                }
                log_likelihood += contribution;
                (mean, covariance)
            };

            predicted_means.push(predicted_mean);
            predicted_covariances.push(predicted_covariance);
            filtered_means.push(filtered_mean.clone());
            filtered_covariances.push(filtered_covariance.clone());
            previous_mean = filtered_mean;
            previous_covariance = filtered_covariance;
        }
        if !log_likelihood.is_finite() {
            return Err(StateSpaceError::NumericalFailure(
                "total log likelihood is not finite".into(),
            ));
        }
        Ok(KalmanFilterResult {
            log_likelihood,
            predicted_means,
            predicted_covariances,
            filtered_means,
            filtered_covariances,
        })
    }

    pub fn smooth(&self, observations: &[f64]) -> Result<KalmanSmootherResult, StateSpaceError> {
        let filter = self.filter(observations)?;
        let count = observations.len();
        if count == 0 {
            return Ok(KalmanSmootherResult {
                filter,
                smoothed_means: Vec::new(),
                smoothed_covariances: Vec::new(),
            });
        }
        let d = self.dimension;
        let mut smoothed_means = filter.filtered_means.clone();
        let mut smoothed_covariances = filter.filtered_covariances.clone();
        for time in (0..count - 1).rev() {
            let filtered_covariance = &filter.filtered_covariances[time];
            let numerator = mat_mul_transpose_right(filtered_covariance, &self.transition, d);
            let factor = cholesky(&filter.predicted_covariances[time + 1], d).map_err(|_| {
                StateSpaceError::NumericalFailure(format!(
                    "predicted covariance at time {} could not be solved",
                    time + 1
                ))
            })?;
            let mut gain = vec![0.0; d * d];
            for row in 0..d {
                let rhs: Vec<f64> = (0..d).map(|column| numerator[row * d + column]).collect();
                let solution = cholesky_solve(&factor, &rhs, d);
                for column in 0..d {
                    gain[row * d + column] = solution[column];
                }
            }

            let mean_delta: Vec<f64> = smoothed_means[time + 1]
                .iter()
                .zip(&filter.predicted_means[time + 1])
                .map(|(smoothed, predicted)| smoothed - predicted)
                .collect();
            let correction = mat_vec(&gain, &mean_delta, d);
            for (entry, delta) in smoothed_means[time].iter_mut().zip(correction) {
                *entry += delta;
            }
            let covariance_delta: Vec<f64> = smoothed_covariances[time + 1]
                .iter()
                .zip(&filter.predicted_covariances[time + 1])
                .map(|(smoothed, predicted)| smoothed - predicted)
                .collect();
            let left = mat_mul(&gain, &covariance_delta, d);
            let covariance_correction = mat_mul_transpose_right(&left, &gain, d);
            add_assign(&mut smoothed_covariances[time], &covariance_correction);
            symmetrize(&mut smoothed_covariances[time], d);
            check_computed(
                "smoothed state",
                &smoothed_means[time],
                &smoothed_covariances[time],
                time,
            )?;
        }
        Ok(KalmanSmootherResult {
            filter,
            smoothed_means,
            smoothed_covariances,
        })
    }

    pub fn forecast(
        &self,
        observations: &[f64],
        steps: usize,
    ) -> Result<ForecastResult, StateSpaceError> {
        let filter = self.filter(observations)?;
        let (mut previous_mean, mut previous_covariance) = match (
            filter.filtered_means.last(),
            filter.filtered_covariances.last(),
        ) {
            (Some(mean), Some(covariance)) => (mean.clone(), covariance.clone()),
            _ => (self.initial_mean.clone(), self.initial_covariance.clone()),
        };
        let d = self.dimension;
        let mut state_means = Vec::with_capacity(steps);
        let mut state_covariances = Vec::with_capacity(steps);
        let mut observation_means = Vec::with_capacity(steps);
        let mut observation_variances = Vec::with_capacity(steps);
        for step in 0..steps {
            let mean = mat_vec(&self.transition, &previous_mean, d);
            let mut covariance = mat_mul_transpose_right(
                &mat_mul(&self.transition, &previous_covariance, d),
                &self.transition,
                d,
            );
            add_assign(&mut covariance, &self.process_covariance);
            symmetrize(&mut covariance, d);
            check_computed("forecast state", &mean, &covariance, step)?;
            let observation_mean = dot(&self.observation, &mean);
            let observation_variance = dot(
                &self.observation,
                &mat_vec(&covariance, &self.observation, d),
            ) + self.observation_variance;
            if !observation_mean.is_finite()
                || !observation_variance.is_finite()
                || observation_variance <= 0.0
            {
                return Err(StateSpaceError::NumericalFailure(format!(
                    "forecast observation moments at step {step} are invalid"
                )));
            }
            state_means.push(mean.clone());
            state_covariances.push(covariance.clone());
            observation_means.push(observation_mean);
            observation_variances.push(observation_variance);
            previous_mean = mean;
            previous_covariance = covariance;
        }

        let square = steps.checked_mul(steps).ok_or_else(|| {
            StateSpaceError::InvalidDimension("forecast horizon is too large".into())
        })?;
        let mut observation_covariance = vec![0.0; square];
        for first in 0..steps {
            let mut cross_covariance = state_covariances[first].clone();
            for second in first..steps {
                if second > first {
                    cross_covariance =
                        mat_mul_transpose_right(&cross_covariance, &self.transition, d);
                }
                let mut covariance = dot(
                    &self.observation,
                    &mat_vec(&cross_covariance, &self.observation, d),
                );
                if first == second {
                    covariance += self.observation_variance;
                }
                if !covariance.is_finite() {
                    return Err(StateSpaceError::NumericalFailure(format!(
                        "joint forecast covariance at ({first}, {second}) is not finite"
                    )));
                }
                observation_covariance[first * steps + second] = covariance;
                observation_covariance[second * steps + first] = covariance;
            }
        }

        let mut cumulative_observation_means = Vec::with_capacity(steps);
        let mut cumulative_observation_variances = Vec::with_capacity(steps);
        let mut cumulative_mean = 0.0;
        let mut cumulative_variance = 0.0;
        for end in 0..steps {
            cumulative_mean += observation_means[end];
            cumulative_variance += observation_covariance[end * steps + end];
            for earlier in 0..end {
                cumulative_variance += 2.0 * observation_covariance[earlier * steps + end];
            }
            if !cumulative_mean.is_finite()
                || !cumulative_variance.is_finite()
                || cumulative_variance <= 0.0
            {
                return Err(StateSpaceError::NumericalFailure(format!(
                    "cumulative forecast moments through step {end} are invalid"
                )));
            }
            cumulative_observation_means.push(cumulative_mean);
            cumulative_observation_variances.push(cumulative_variance);
        }
        Ok(ForecastResult {
            state_means,
            state_covariances,
            observation_means,
            observation_variances,
            observation_covariance,
            cumulative_observation_means,
            cumulative_observation_variances,
        })
    }

    /// Draw the pre-observation initial state and all filtered states from the
    /// joint smoothing distribution. This is crate-private because callers
    /// must still provide an outer parameter sampler to obtain Bayesian fits.
    pub(crate) fn sample_states_ffbs<R: Rng + ?Sized>(
        &self,
        observations: &[f64],
        rng: &mut R,
    ) -> Result<Vec<Vec<f64>>, StateSpaceError> {
        let filter = self.filter(observations)?;
        let count = observations.len();
        let d = self.dimension;
        let mut filtered_means = Vec::with_capacity(count + 1);
        let mut filtered_covariances = Vec::with_capacity(count + 1);
        filtered_means.push(self.initial_mean.clone());
        filtered_means.extend(filter.filtered_means.iter().cloned());
        filtered_covariances.push(self.initial_covariance.clone());
        filtered_covariances.extend(filter.filtered_covariances.iter().cloned());

        let mut states = vec![vec![0.0; d]; count + 1];
        states[count] = sample_multivariate_normal(
            &filtered_means[count],
            &filtered_covariances[count],
            d,
            rng,
        )?;

        for index in (0..count).rev() {
            let filtered_covariance = &filtered_covariances[index];
            let numerator = mat_mul_transpose_right(filtered_covariance, &self.transition, d);
            let prediction_factor =
                cholesky(&filter.predicted_covariances[index], d).map_err(|_| {
                    StateSpaceError::NumericalFailure(format!(
                        "predicted covariance at time {index} could not be factored for FFBS"
                    ))
                })?;
            let mut gain = vec![0.0; d * d];
            for row in 0..d {
                let rhs = &numerator[row * d..(row + 1) * d];
                let solution = cholesky_solve(&prediction_factor, rhs, d);
                gain[row * d..(row + 1) * d].copy_from_slice(&solution);
            }

            let delta: Vec<f64> = states[index + 1]
                .iter()
                .zip(&filter.predicted_means[index])
                .map(|(sampled, predicted)| sampled - predicted)
                .collect();
            let correction = mat_vec(&gain, &delta, d);
            let conditional_mean: Vec<f64> = filtered_means[index]
                .iter()
                .zip(correction)
                .map(|(mean, correction)| mean + correction)
                .collect();

            // Stable Joseph-style form for P - J P_pred J':
            // (I - J T) P (I - J T)' + J Q J'.
            let mut residual_transition = identity(d);
            let gain_transition = mat_mul(&gain, &self.transition, d);
            for (entry, subtraction) in residual_transition.iter_mut().zip(gain_transition) {
                *entry -= subtraction;
            }
            let propagated = mat_mul_transpose_right(
                &mat_mul(&residual_transition, filtered_covariance, d),
                &residual_transition,
                d,
            );
            let process_contribution =
                mat_mul_transpose_right(&mat_mul(&gain, &self.process_covariance, d), &gain, d);
            let mut conditional_covariance = propagated;
            add_assign(&mut conditional_covariance, &process_contribution);
            symmetrize(&mut conditional_covariance, d);
            states[index] =
                sample_multivariate_normal(&conditional_mean, &conditional_covariance, d, rng)?;
        }
        Ok(states)
    }
}

fn check_len(name: &str, actual: usize, expected: usize) -> Result<(), StateSpaceError> {
    if actual != expected {
        return Err(StateSpaceError::InvalidDimension(format!(
            "{name} has {actual} entries; expected {expected}"
        )));
    }
    Ok(())
}

fn check_finite(name: &str, values: &[f64]) -> Result<(), StateSpaceError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(StateSpaceError::NonFinite(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_observations(values: &[f64]) -> Result<(), StateSpaceError> {
    if values.iter().any(|value| value.is_infinite()) {
        return Err(StateSpaceError::NonFinite(
            "observations may contain finite values or NaN for missing values, but not infinity"
                .into(),
        ));
    }
    Ok(())
}

fn check_symmetric(name: &str, matrix: &[f64], d: usize) -> Result<(), StateSpaceError> {
    for i in 0..d {
        for j in 0..i {
            let a = matrix[i * d + j];
            let b = matrix[j * d + i];
            let scale = 1.0_f64.max(a.abs()).max(b.abs());
            if (a - b).abs() > SYMMETRY_TOLERANCE * scale {
                return Err(StateSpaceError::NotSymmetric(format!(
                    "{name} differs at ({i}, {j}) and ({j}, {i})"
                )));
            }
        }
    }
    Ok(())
}

fn check_computed(
    name: &str,
    mean: &[f64],
    covariance: &[f64],
    index: usize,
) -> Result<(), StateSpaceError> {
    if mean
        .iter()
        .chain(covariance)
        .any(|value| !value.is_finite())
    {
        return Err(StateSpaceError::NumericalFailure(format!(
            "{name} at index {index} contains a non-finite value"
        )));
    }
    Ok(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn identity(d: usize) -> Vec<f64> {
    let mut result = vec![0.0; d * d];
    for i in 0..d {
        result[i * d + i] = 1.0;
    }
    result
}

fn mat_vec(matrix: &[f64], vector: &[f64], d: usize) -> Vec<f64> {
    (0..d)
        .map(|row| dot(&matrix[row * d..(row + 1) * d], vector))
        .collect()
}

fn mat_mul(left: &[f64], right: &[f64], d: usize) -> Vec<f64> {
    let mut result = vec![0.0; d * d];
    for i in 0..d {
        for k in 0..d {
            let value = left[i * d + k];
            for j in 0..d {
                result[i * d + j] += value * right[k * d + j];
            }
        }
    }
    result
}

fn mat_mul_transpose_right(left: &[f64], right: &[f64], d: usize) -> Vec<f64> {
    let mut result = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                result[i * d + j] += left[i * d + k] * right[j * d + k];
            }
        }
    }
    result
}

fn add_assign(target: &mut [f64], addition: &[f64]) {
    for (target, addition) in target.iter_mut().zip(addition) {
        *target += addition;
    }
}

fn symmetrize(matrix: &mut [f64], d: usize) {
    for i in 0..d {
        for j in 0..i {
            let average = 0.5 * (matrix[i * d + j] + matrix[j * d + i]);
            matrix[i * d + j] = average;
            matrix[j * d + i] = average;
        }
    }
}

fn cholesky(matrix: &[f64], d: usize) -> Result<Vec<f64>, ()> {
    let mut factor = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut value = matrix[i * d + j];
            for k in 0..j {
                value -= factor[i * d + k] * factor[j * d + k];
            }
            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return Err(());
                }
                factor[i * d + j] = value.sqrt();
            } else {
                factor[i * d + j] = value / factor[j * d + j];
            }
        }
    }
    Ok(factor)
}

/// Cholesky-like validation for positive-semidefinite covariance matrices.
/// A zero pivot is valid only when the corresponding residual off-diagonal
/// entries are also numerically zero.
fn positive_semidefinite_factor(matrix: &[f64], d: usize) -> Result<Vec<f64>, ()> {
    let mut factor = vec![0.0; d * d];
    let scale = matrix.iter().map(|value| value.abs()).fold(1.0, f64::max);
    let tolerance = 1e-12 * scale;
    for i in 0..d {
        for j in 0..=i {
            let mut value = matrix[i * d + j];
            for k in 0..j {
                value -= factor[i * d + k] * factor[j * d + k];
            }
            if i == j {
                if !value.is_finite() || value < -tolerance {
                    return Err(());
                }
                factor[i * d + j] = value.max(0.0).sqrt();
            } else if factor[j * d + j] > tolerance.sqrt() {
                factor[i * d + j] = value / factor[j * d + j];
            } else if value.abs() > tolerance {
                return Err(());
            }
        }
    }
    Ok(factor)
}

fn sample_multivariate_normal<R: Rng + ?Sized>(
    mean: &[f64],
    covariance: &[f64],
    d: usize,
    rng: &mut R,
) -> Result<Vec<f64>, StateSpaceError> {
    let factor = positive_semidefinite_factor(covariance, d).map_err(|_| {
        StateSpaceError::NumericalFailure(
            "conditional smoothing covariance is not positive semidefinite".into(),
        )
    })?;
    let standard: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let mut draw = mean.to_vec();
    for row in 0..d {
        for column in 0..=row {
            draw[row] += factor[row * d + column] * standard[column];
        }
    }
    if draw.iter().any(|value| !value.is_finite()) {
        return Err(StateSpaceError::NumericalFailure(
            "FFBS produced a non-finite state draw".into(),
        ));
    }
    Ok(draw)
}

fn cholesky_solve(factor: &[f64], rhs: &[f64], d: usize) -> Vec<f64> {
    let mut result = rhs.to_vec();
    for i in 0..d {
        for j in 0..i {
            result[i] -= factor[i * d + j] * result[j];
        }
        result[i] /= factor[i * d + i];
    }
    for i in (0..d).rev() {
        for j in i + 1..d {
            result[i] -= factor[j * d + i] * result[j];
        }
        result[i] /= factor[i * d + i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn scalar_filter_matches_closed_form_update() {
        let model = LinearGaussianStateSpace::local_level(1.0, 2.0, 0.0, 3.0).unwrap();
        let result = model.filter(&[2.0]).unwrap();
        // Prediction: a=0, P=4; S=6, K=2/3.
        assert_close(result.predicted_means[0][0], 0.0);
        assert_close(result.predicted_covariances[0][0], 4.0);
        assert_close(result.filtered_means[0][0], 4.0 / 3.0);
        assert_close(result.filtered_covariances[0][0], 4.0 / 3.0);
        assert_close(
            result.log_likelihood,
            -0.5 * (LOG_2_PI + 6.0_f64.ln() + 4.0 / 6.0),
        );
    }

    #[test]
    fn missing_observation_is_prediction_only_and_adds_no_likelihood() {
        let model = LinearGaussianStateSpace::local_level(1.0, 2.0, 3.0, 4.0).unwrap();
        let result = model.filter(&[f64::NAN]).unwrap();
        assert_eq!(result.log_likelihood, 0.0);
        assert_eq!(result.filtered_means[0], result.predicted_means[0]);
        assert_eq!(
            result.filtered_covariances[0],
            result.predicted_covariances[0]
        );
        assert_close(result.filtered_means[0][0], 3.0);
        assert_close(result.filtered_covariances[0][0], 5.0);
    }

    #[test]
    fn scalar_smoother_matches_closed_form_backward_update() {
        let model = LinearGaussianStateSpace::local_level(1.0, 1.0, 0.0, 1.0).unwrap();
        let result = model.smooth(&[1.0, 2.0]).unwrap();
        // t0 filter: m=2/3, P=2/3. t1 prediction P=5/3; smoother gain=2/5.
        assert_close(result.smoothed_means[0][0], 1.0);
        assert_close(result.smoothed_covariances[0][0], 0.5);
        assert_eq!(result.smoothed_means[1], result.filter.filtered_means[1]);
    }

    #[test]
    fn scalar_forecast_propagates_state_and_observation_variance() {
        let model = LinearGaussianStateSpace::local_level(1.0, 2.0, 0.0, 3.0).unwrap();
        let result = model.forecast(&[], 2).unwrap();
        assert_eq!(result.observation_means, vec![0.0, 0.0]);
        assert_close(result.state_covariances[0][0], 4.0);
        assert_close(result.state_covariances[1][0], 5.0);
        assert_close(result.observation_variances[0], 6.0);
        assert_close(result.observation_variances[1], 7.0);
        assert_eq!(result.observation_covariance, vec![6.0, 4.0, 4.0, 7.0]);
        assert_eq!(result.cumulative_observation_means, vec![0.0, 0.0]);
        assert_close(result.cumulative_observation_variances[0], 6.0);
        // Var(y1 + y2) = 6 + 7 + 2 * Cov(y1, y2), where Cov=4.
        assert_close(result.cumulative_observation_variances[1], 21.0);
    }

    #[test]
    fn seasonal_local_level_repeats_a_sum_to_zero_cycle() {
        let effects = vec![1.0, -0.5, -0.25, -0.25];
        let model = LinearGaussianStateSpace::seasonal_local_level(
            4,
            0.0,
            0.0,
            0.1,
            10.0,
            effects.clone(),
            1.0,
            1.0,
        )
        .unwrap();

        assert_eq!(model.dimension(), 4);
        assert_eq!(model.process_covariance, vec![0.0; 16]);
        let forecast = model.forecast(&[], 8).unwrap();
        let expected: Vec<f64> = effects
            .iter()
            .cycle()
            .take(8)
            .map(|effect| 10.0 + effect)
            .collect();
        assert_eq!(forecast.observation_means, expected);
        assert!(forecast
            .observation_covariance
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn seasonal_local_level_validates_period_effects_and_variances() {
        assert!(matches!(
            LinearGaussianStateSpace::seasonal_local_level(
                1,
                1.0,
                1.0,
                1.0,
                0.0,
                vec![0.0],
                1.0,
                1.0,
            ),
            Err(StateSpaceError::InvalidParameter(_))
        ));
        assert!(matches!(
            LinearGaussianStateSpace::seasonal_local_level(
                4,
                1.0,
                1.0,
                1.0,
                0.0,
                vec![1.0, 0.0, 0.0, 0.0],
                1.0,
                1.0,
            ),
            Err(StateSpaceError::InvalidParameter(_))
        ));
        assert!(matches!(
            LinearGaussianStateSpace::seasonal_local_level(
                4,
                -1.0,
                1.0,
                1.0,
                0.0,
                vec![0.0; 4],
                1.0,
                1.0,
            ),
            Err(StateSpaceError::InvalidVariance(_))
        ));
    }

    #[test]
    fn seasonal_ffbs_preserves_deterministic_shifts_and_matches_terminal_moments() {
        let model = LinearGaussianStateSpace::seasonal_local_level(
            4,
            0.08,
            0.03,
            0.2,
            5.0,
            vec![1.0, -0.5, -0.25, -0.25],
            2.0,
            1.0,
        )
        .unwrap();
        let observations = [6.0, 4.7, f64::NAN, 4.8, 6.1, 4.5, 4.9, 4.8];
        let smoother = model.smooth(&observations).unwrap();
        let expected_mean = smoother.smoothed_means.last().unwrap();
        let expected_covariance = smoother.smoothed_covariances.last().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(91);
        let draws = 6000;
        let mut sums = [0.0; 4];
        let mut products = [0.0; 16];
        for _ in 0..draws {
            let states = model.sample_states_ffbs(&observations, &mut rng).unwrap();
            for pair in states.windows(2) {
                assert!((pair[1][2] - pair[0][1]).abs() < 1e-8);
                assert!((pair[1][3] - pair[0][2]).abs() < 1e-8);
            }
            let terminal = states.last().unwrap();
            for row in 0..4 {
                sums[row] += terminal[row];
                for column in 0..4 {
                    products[row * 4 + column] += terminal[row] * terminal[column];
                }
            }
        }
        let sampled_mean: Vec<f64> = sums.iter().map(|sum| sum / draws as f64).collect();
        for index in 0..4 {
            assert!((sampled_mean[index] - expected_mean[index]).abs() < 0.04);
        }
        for row in 0..4 {
            for column in 0..4 {
                let sampled_covariance = products[row * 4 + column] / draws as f64
                    - sampled_mean[row] * sampled_mean[column];
                assert!(
                    (sampled_covariance - expected_covariance[row * 4 + column]).abs() < 0.05,
                    "covariance ({row}, {column}) differs: {sampled_covariance} vs {}",
                    expected_covariance[row * 4 + column]
                );
            }
        }
    }

    #[test]
    fn stationary_ar1_uses_the_stationary_initial_distribution() {
        let model = LinearGaussianStateSpace::stationary_ar1(0.8, 0.36, 0.25).unwrap();
        assert_eq!(model.dimension, 1);
        assert_eq!(model.transition, vec![0.8]);
        assert_eq!(model.observation, vec![1.0]);
        assert_eq!(model.process_covariance, vec![0.36]);
        assert_eq!(model.observation_variance, 0.25);
        assert_eq!(model.initial_mean, vec![0.0]);
        assert_close(model.initial_covariance[0], 1.0);

        let result = model.forecast(&[], 3).unwrap();
        assert_eq!(result.observation_means, vec![0.0; 3]);
        for covariance in result.state_covariances {
            assert_close(covariance[0], 1.0);
        }
        for variance in result.observation_variances {
            assert_close(variance, 1.25);
        }
    }

    #[test]
    fn stationary_ar1_rejects_nonstationary_or_nonfinite_coefficients() {
        for coefficient in [-1.0, 1.0, -1.01, 1.01] {
            assert!(matches!(
                LinearGaussianStateSpace::stationary_ar1(coefficient, 1.0, 1.0),
                Err(StateSpaceError::InvalidParameter(_))
            ));
        }
        for coefficient in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert!(matches!(
                LinearGaussianStateSpace::stationary_ar1(coefficient, 1.0, 1.0),
                Err(StateSpaceError::NonFinite(_))
            ));
        }
    }

    #[test]
    fn stationary_ar1_rejects_invalid_or_unrepresentable_variances() {
        for process_variance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                LinearGaussianStateSpace::stationary_ar1(0.5, process_variance, 1.0),
                Err(StateSpaceError::InvalidVariance(_))
            ));
        }
        for observation_variance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                LinearGaussianStateSpace::stationary_ar1(0.5, 1.0, observation_variance),
                Err(StateSpaceError::InvalidVariance(_))
            ));
        }

        assert!(matches!(
            LinearGaussianStateSpace::stationary_ar1(1.0 - f64::EPSILON, f64::MAX, 1.0,),
            Err(StateSpaceError::InvalidVariance(_))
        ));
    }

    #[test]
    fn validation_rejects_bad_inputs_without_panicking() {
        assert!(matches!(
            LinearGaussianStateSpace::new(
                2,
                vec![1.0; 3],
                vec![1.0, 0.0],
                vec![1.0; 4],
                1.0,
                vec![0.0; 2],
                vec![1.0; 4]
            ),
            Err(StateSpaceError::InvalidDimension(_))
        ));
        assert!(matches!(
            LinearGaussianStateSpace::new(
                2,
                vec![1.0, 0.0, 0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0, 0.0, 1.0],
                1.0,
                vec![0.0; 2],
                vec![1.0, 0.0, 0.0, 1.0]
            ),
            Err(StateSpaceError::NotSymmetric(_))
        ));
        LinearGaussianStateSpace::local_level(0.0, 1.0, 0.0, 1.0)
            .expect("deterministic state evolution has a valid semidefinite covariance");
        assert!(matches!(
            LinearGaussianStateSpace::new(
                2,
                vec![1.0, 0.0, 0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 2.0, 2.0, 1.0],
                1.0,
                vec![0.0; 2],
                vec![1.0, 0.0, 0.0, 1.0]
            ),
            Err(StateSpaceError::NotPositiveSemidefinite(_))
        ));
        let model = LinearGaussianStateSpace::local_level(1.0, 1.0, 0.0, 1.0).unwrap();
        assert!(matches!(
            model.filter(&[f64::INFINITY]),
            Err(StateSpaceError::NonFinite(_))
        ));
    }

    #[test]
    fn numerical_overflow_is_reported_as_an_error() {
        let model = LinearGaussianStateSpace::new(
            1,
            vec![f64::MAX],
            vec![1.0],
            vec![1.0],
            1.0,
            vec![1.0],
            vec![1.0],
        )
        .unwrap();
        assert!(matches!(
            model.filter(&[0.0]),
            Err(StateSpaceError::NumericalFailure(_))
        ));
    }
}
