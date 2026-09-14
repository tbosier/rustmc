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
    observation_rows: Option<Vec<Vec<f64>>>,
    process_covariance: Vec<f64>,
    diagonal_process: bool,
    observation_variance: f64,
    observation_variances: Option<Vec<f64>>,
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

        let diagonal_process = process_covariance
            .iter()
            .enumerate()
            .all(|(i, &v)| i / dimension == i % dimension || v == 0.0);
        Ok(Self {
            dimension,
            transition,
            observation,
            observation_rows: None,
            process_covariance,
            diagonal_process,
            observation_variance,
            observation_variances: None,
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

    /// Set one finite observation row per training time. Missing observations
    /// retain their rows. Forecasting then requires explicit future rows.
    pub fn with_observation_rows(mut self, rows: Vec<Vec<f64>>) -> Result<Self, StateSpaceError> {
        self.validate_rows(&rows, rows.len())?;
        self.observation_rows = Some(rows);
        Ok(self)
    }

    /// Set per-time Gaussian noise variances, e.g. conditional Student-t mixtures.
    pub fn with_observation_variances(
        mut self,
        variances: Vec<f64>,
    ) -> Result<Self, StateSpaceError> {
        if variances.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(StateSpaceError::InvalidVariance(
                "observation variances must be finite and positive".into(),
            ));
        }
        self.observation_variances = Some(variances);
        Ok(self)
    }

    pub(crate) fn has_observation_variances(&self) -> bool {
        self.observation_variances.is_some()
    }

    pub(crate) fn process_covariance(&self) -> &[f64] {
        &self.process_covariance
    }

    /// Simulate the complete next state, including fixed correlated innovations.
    pub fn simulate_transition<R: Rng + ?Sized>(
        &self,
        state: &[f64],
        rng: &mut R,
    ) -> Result<Vec<f64>, StateSpaceError> {
        check_len("state", state.len(), self.dimension)?;
        check_finite("state", state)?;
        let mean = mat_vec(&self.transition, state, self.dimension);
        // Preset innovations are diagonal, often with many deterministic shift
        // coordinates. Keep their simulation quadratic in dimension (transition
        // multiplication), rather than factoring a dense matrix at each horizon.
        if self.diagonal_process {
            let draw: Vec<f64> = mean
                .iter()
                .enumerate()
                .map(|(i, &m)| {
                    let z: f64 = StandardNormal.sample(rng);
                    m + self.process_covariance[i * self.dimension + i].sqrt() * z
                })
                .collect();
            check_finite("simulated state", &draw)?;
            Ok(draw)
        } else {
            sample_multivariate_normal(&mean, &self.process_covariance, self.dimension, rng)
        }
    }

    pub fn simulate_initial<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<Vec<f64>, StateSpaceError> {
        sample_multivariate_normal(
            &self.initial_mean,
            &self.initial_covariance,
            self.dimension,
            rng,
        )
    }

    fn validate_rows(&self, rows: &[Vec<f64>], count: usize) -> Result<(), StateSpaceError> {
        check_len("observation row count", rows.len(), count)?;
        for row in rows {
            check_len("observation row", row.len(), self.dimension)?;
            check_finite("observation rows", row)?;
        }
        Ok(())
    }

    pub(crate) fn transition(&self) -> &[f64] {
        &self.transition
    }
    pub(crate) fn observation(&self) -> &[f64] {
        &self.observation
    }
    pub(crate) fn set_variances(&mut self, indices: &[usize], variances: &[f64], observation: f64) {
        for (&index, &variance) in indices.iter().zip(variances) {
            self.process_covariance[index * self.dimension + index] = variance;
        }
        self.observation_variance = observation;
    }

    /// Augment a structural model with static uncertain coefficients. The
    /// coefficient block has identity transition and exactly zero process noise.
    pub fn with_static_regression(
        &self,
        design: &[Vec<f64>],
        mean: &[f64],
        covariance: &[f64],
    ) -> Result<Self, StateSpaceError> {
        if self.observation_rows.is_some() {
            return Err(StateSpaceError::InvalidParameter(
                "static regression augmentation requires a constant structural observation row"
                    .into(),
            ));
        }
        let p = mean.len();
        if p == 0 {
            return Err(StateSpaceError::InvalidDimension(
                "coefficient prior must not be empty".into(),
            ));
        }
        check_len(
            "coefficient covariance",
            covariance.len(),
            p.checked_mul(p)
                .ok_or_else(|| StateSpaceError::InvalidDimension("too many coefficients".into()))?,
        )?;
        let n = self.dimension + p;
        let mut transition = vec![0.0; n * n];
        let mut process = vec![0.0; n * n];
        let mut initial = vec![0.0; n * n];
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                transition[i * n + j] = self.transition[i * self.dimension + j];
                process[i * n + j] = self.process_covariance[i * self.dimension + j];
                initial[i * n + j] = self.initial_covariance[i * self.dimension + j];
            }
        }
        for i in 0..p {
            transition[(i + self.dimension) * n + i + self.dimension] = 1.0;
            for j in 0..p {
                initial[(i + self.dimension) * n + j + self.dimension] = covariance[i * p + j];
            }
        }
        let mut initial_mean = self.initial_mean.clone();
        initial_mean.extend_from_slice(mean);
        let mut observation = self.observation.clone();
        observation.resize(n, 0.0);
        let rows = design
            .iter()
            .map(|row| {
                check_len("exog feature count", row.len(), p)?;
                let mut full = self.observation.clone();
                full.extend_from_slice(row);
                Ok(full)
            })
            .collect::<Result<Vec<_>, StateSpaceError>>()?;
        let mut model = Self::new(
            n,
            transition,
            observation,
            process,
            self.observation_variance,
            initial_mean,
            initial,
        )?
        .with_observation_rows(rows)?;
        model.observation_variances = self.observation_variances.clone();
        Ok(model)
    }

    pub fn filter(&self, observations: &[f64]) -> Result<KalmanFilterResult, StateSpaceError> {
        validate_observations(observations)?;
        if let Some(rows) = &self.observation_rows {
            self.validate_rows(rows, observations.len())?;
        }
        if let Some(variances) = &self.observation_variances {
            check_len(
                "observation variance count",
                variances.len(),
                observations.len(),
            )?;
        }
        let d = self.dimension;
        let mut previous_mean = self.initial_mean.clone();
        let mut previous_covariance = self.initial_covariance.clone();
        let mut predicted_means = Vec::with_capacity(observations.len());
        let mut predicted_covariances = Vec::with_capacity(observations.len());
        let mut filtered_means = Vec::with_capacity(observations.len());
        let mut filtered_covariances = Vec::with_capacity(observations.len());
        let mut log_likelihood = 0.0;

        for (time, &value) in observations.iter().enumerate() {
            let noise_variance = self
                .observation_variances
                .as_ref()
                .map_or(self.observation_variance, |v| v[time]);
            let observation = self
                .observation_rows
                .as_ref()
                .map_or(self.observation.as_slice(), |rows| rows[time].as_slice());
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
                let ph = mat_vec(&predicted_covariance, observation, d);
                let innovation_variance = dot(observation, &ph) + noise_variance;
                if !innovation_variance.is_finite() || innovation_variance <= 0.0 {
                    return Err(StateSpaceError::NumericalFailure(format!(
                        "innovation variance at time {time} is not finite and positive"
                    )));
                }
                let predicted_observation = dot(observation, &predicted_mean);
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
                        update[i * d + j] -= gain[i] * observation[j];
                    }
                }
                let left = mat_mul(&update, &predicted_covariance, d);
                let mut covariance = mat_mul_transpose_right(&left, &update, d);
                for i in 0..d {
                    for j in 0..d {
                        covariance[i * d + j] += gain[i] * noise_variance * gain[j];
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
        let (backward, mut smoothed_factor) = self.backward_parameters(observations)?;
        let mut smoothed_means = filter.filtered_means.clone();
        let mut smoothed_covariances = filter.filtered_covariances.clone();
        smoothed_covariances[count - 1] = root_covariance(&smoothed_factor, d, d);
        for time in (0..count - 1).rev() {
            let gain = &backward[time + 1].gain;

            let mean_delta: Vec<f64> = smoothed_means[time + 1]
                .iter()
                .zip(&filter.predicted_means[time + 1])
                .map(|(smoothed, predicted)| smoothed - predicted)
                .collect();
            let correction = mat_vec(gain, &mean_delta, d);
            for (entry, delta) in smoothed_means[time].iter_mut().zip(correction) {
                *entry += delta;
            }
            // P(x_t | all y) = P(x_t | x_{t+1}, past y)
            //                     + J P(x_{t+1} | all y) J'.
            // Avoid subtracting the predicted covariance from the smoothed
            // covariance: that can erase a small, valid posterior variance.
            let propagated = mat_mul(gain, &smoothed_factor, d);
            let mut combined = vec![0.0; d * 3 * d];
            for row in 0..d {
                combined[row * 3 * d..row * 3 * d + 2 * d]
                    .copy_from_slice(&backward[time + 1].factor[row * 2 * d..(row + 1) * 2 * d]);
                combined[row * 3 * d + 2 * d..(row + 1) * 3 * d]
                    .copy_from_slice(&propagated[row * d..(row + 1) * d]);
            }
            smoothed_factor = orthogonal_rows(&combined, d, 3 * d)
                .map_err(|_| {
                    StateSpaceError::NumericalFailure("smoothed covariance root failed".into())
                })?
                .factor;
            smoothed_covariances[time] = root_covariance(&smoothed_factor, d, d);
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
        if self.observation_rows.is_some() {
            return Err(StateSpaceError::InvalidParameter(
                "future observation rows are required for a time-varying design".into(),
            ));
        }
        self.forecast_with_observation_rows(observations, &vec![self.observation.clone(); steps])
    }

    /// Forecast with a row for each future step, continuing after the final
    /// training time. Both rows enter each cross-horizon covariance.
    pub fn forecast_with_observation_rows(
        &self,
        observations: &[f64],
        future_rows: &[Vec<f64>],
    ) -> Result<ForecastResult, StateSpaceError> {
        let steps = future_rows.len();
        self.validate_rows(future_rows, steps)?;
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
        for (step, observation) in future_rows.iter().enumerate() {
            let mean = mat_vec(&self.transition, &previous_mean, d);
            let mut covariance = mat_mul_transpose_right(
                &mat_mul(&self.transition, &previous_covariance, d),
                &self.transition,
                d,
            );
            add_assign(&mut covariance, &self.process_covariance);
            symmetrize(&mut covariance, d);
            check_computed("forecast state", &mean, &covariance, step)?;
            let observation_mean = dot(observation, &mean);
            let observation_variance =
                dot(observation, &mat_vec(&covariance, observation, d)) + self.observation_variance;
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
                    &future_rows[first],
                    &mat_vec(&cross_covariance, &future_rows[second], d),
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

    /// Propagate covariance roots so rank is determined before squaring them.
    /// Covariance roundoff is O(epsilon), which cannot distinguish a true small
    /// eigenvalue from a lost deterministic direction. In root coordinates the
    /// same small eigenvalue has O(sqrt(epsilon)) amplitude and is retained.
    fn backward_parameters(
        &self,
        observations: &[f64],
    ) -> Result<(Vec<BackwardConditional>, Vec<f64>), StateSpaceError> {
        let d = self.dimension;
        let width = 2 * d;
        let failure =
            || StateSpaceError::NumericalFailure("square-root backward conditioning failed".into());
        let mut filtered = cholesky(&self.initial_covariance, d).map_err(|_| failure())?;
        let process = process_root(&self.process_covariance, d).map_err(|_| failure())?;
        let mut backward = Vec::with_capacity(observations.len());
        for (time, value) in observations.iter().enumerate() {
            let propagated = mat_mul(&self.transition, &filtered, d);
            let mut joint = vec![0.0; d * width];
            for i in 0..d {
                joint[i * width..i * width + d].copy_from_slice(&propagated[i * d..(i + 1) * d]);
                joint[i * width + d..(i + 1) * width].copy_from_slice(&process[i * d..(i + 1) * d]);
            }
            let roots = orthogonal_rows(&joint, d, width).map_err(|_| failure())?;
            let rank = roots.indices.len();
            let mut projection = vec![0.0; d * rank];
            let mut residual = vec![0.0; d * width];
            for i in 0..d {
                residual[i * width..i * width + d].copy_from_slice(&filtered[i * d..(i + 1) * d]);
                for j in 0..rank {
                    let coefficient: f64 = (0..d)
                        .map(|k| filtered[i * d + k] * roots.basis[j * width + k])
                        .sum();
                    projection[i * rank + j] = coefficient;
                    for k in 0..width {
                        residual[i * width + k] -= coefficient * roots.basis[j * width + k];
                    }
                }
            }
            let mut gain = vec![0.0; d * d];
            for row in 0..d {
                // The independent rows of the root form a lower triangular
                // matrix in pivot order. Solve its transpose, not P_pred.
                let mut solution = projection[row * rank..(row + 1) * rank].to_vec();
                for i in (0..rank).rev() {
                    for j in i + 1..rank {
                        solution[i] -= roots.factor[roots.indices[j] * d + i] * solution[j];
                    }
                    solution[i] /= roots.factor[roots.indices[i] * d + i];
                    gain[row * d + roots.indices[i]] = solution[i];
                }
            }
            // An invertible identity transition with no innovations conveys
            // the entire state exactly, including arbitrarily small modes.
            if self.transition == identity(d) && self.process_covariance.iter().all(|x| *x == 0.0) {
                gain = identity(d);
                residual.fill(0.0);
            }
            let covariance = root_covariance(&residual, d, width);
            check_computed("backward conditional", &gain, &covariance, time)?;
            backward.push(BackwardConditional {
                gain,
                factor: residual,
            });
            filtered = roots.factor;
            if value.is_finite() {
                let h = self
                    .observation_rows
                    .as_ref()
                    .map_or(self.observation.as_slice(), |rows| rows[time].as_slice());
                let noise = self
                    .observation_variances
                    .as_ref()
                    .map_or(self.observation_variance, |v| v[time]);
                let u: Vec<f64> = (0..d)
                    .map(|j| (0..d).map(|i| h[i] * filtered[i * d + j]).sum())
                    .collect();
                let variance = noise + u.iter().map(|x| x * x).sum::<f64>();
                let gain: Vec<f64> = (0..d)
                    .map(|i| (0..d).map(|j| filtered[i * d + j] * u[j]).sum::<f64>() / variance)
                    .collect();
                let mut update = vec![0.0; d * (d + 1)];
                for i in 0..d {
                    for j in 0..d {
                        update[i * (d + 1) + j] = filtered[i * d + j] - gain[i] * u[j];
                    }
                    update[i * (d + 1) + d] = gain[i] * noise.sqrt();
                }
                filtered = orthogonal_rows(&update, d, d + 1)
                    .map_err(|_| failure())?
                    .factor;
            }
        }
        Ok((backward, filtered))
    }

    /// Draw the initial state and all observation-time states jointly.
    pub(crate) fn sample_states_ffbs<R: Rng + ?Sized>(
        &self,
        observations: &[f64],
        rng: &mut R,
    ) -> Result<Vec<Vec<f64>>, StateSpaceError> {
        let filter = self.filter(observations)?;
        let count = observations.len();
        let d = self.dimension;
        let mut filtered_means = Vec::with_capacity(count + 1);
        filtered_means.push(self.initial_mean.clone());
        filtered_means.extend(filter.filtered_means.iter().cloned());

        let (backward, terminal_factor) = self.backward_parameters(observations)?;
        let mut states = vec![vec![0.0; d]; count + 1];
        states[count] = sample_from_factor(&filtered_means[count], &terminal_factor, d, d, rng)?;

        for index in (0..count).rev() {
            let conditional = &backward[index];
            let gain = &conditional.gain;

            let delta: Vec<f64> = states[index + 1]
                .iter()
                .zip(&filter.predicted_means[index])
                .map(|(sampled, predicted)| sampled - predicted)
                .collect();
            let correction = mat_vec(gain, &delta, d);
            let conditional_mean: Vec<f64> = filtered_means[index]
                .iter()
                .zip(correction)
                .map(|(mean, correction)| mean + correction)
                .collect();

            states[index] =
                sample_from_factor(&conditional_mean, &conditional.factor, d, 2 * d, rng)?;
        }
        Ok(states)
    }
}

struct BackwardConditional {
    gain: Vec<f64>,
    factor: Vec<f64>,
}

struct OrthogonalRows {
    factor: Vec<f64>,
    basis: Vec<f64>,
    indices: Vec<usize>,
}

fn root_covariance(root: &[f64], d: usize, width: usize) -> Vec<f64> {
    let mut covariance = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let value = (0..width)
                .map(|k| root[i * width + k] * root[j * width + k])
                .sum();
            covariance[i * d + j] = value;
            covariance[j * d + i] = value;
        }
    }
    covariance
}

fn sample_from_factor<R: Rng + ?Sized>(
    mean: &[f64],
    root: &[f64],
    d: usize,
    width: usize,
    rng: &mut R,
) -> Result<Vec<f64>, StateSpaceError> {
    let z: Vec<f64> = (0..width).map(|_| StandardNormal.sample(rng)).collect();
    let draw: Vec<f64> = (0..d)
        .map(|i| mean[i] + (0..width).map(|j| root[i * width + j] * z[j]).sum::<f64>())
        .collect();
    if draw.iter().any(|x| !x.is_finite()) {
        return Err(StateSpaceError::NumericalFailure(
            "FFBS produced a non-finite state draw".into(),
        ));
    }
    Ok(draw)
}

/// Rank-revealing modified Gram-Schmidt on equilibrated root rows. A second
/// orthogonalization pass avoids mistaking cancellation for an extra direction.
fn orthogonal_rows(matrix: &[f64], d: usize, width: usize) -> Result<OrthogonalRows, ()> {
    let mut scales = vec![0.0_f64; d];
    let mut residual = matrix.to_vec();
    for i in 0..d {
        for &x in &matrix[i * width..(i + 1) * width] {
            if !x.is_finite() {
                return Err(());
            }
            scales[i] = scales[i].max(x.abs());
        }
        if scales[i] > 0.0 {
            for x in &mut residual[i * width..(i + 1) * width] {
                *x /= scales[i];
            }
        }
    }
    let tolerance = 64.0 * f64::EPSILON * (d + width) as f64;
    let mut remaining: Vec<usize> = (0..d).collect();
    let mut indices = Vec::new();
    let mut basis = vec![0.0; d * width];
    let mut factor = vec![0.0; d * d];
    for column in 0..d {
        let norm = |i: usize| {
            residual[i * width..(i + 1) * width]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        };
        let pivot = *remaining
            .iter()
            .max_by(|&&a, &&b| norm(a).total_cmp(&norm(b)))
            .ok_or(())?;
        let length = norm(pivot);
        if length <= tolerance {
            break;
        }
        for k in 0..width {
            basis[column * width + k] = residual[pivot * width + k] / length;
        }
        indices.push(pivot);
        for &row in &remaining {
            let mut coefficient = 0.0;
            for _ in 0..2 {
                let correction: f64 = (0..width)
                    .map(|k| residual[row * width + k] * basis[column * width + k])
                    .sum();
                coefficient += correction;
                for k in 0..width {
                    residual[row * width + k] -= correction * basis[column * width + k];
                }
            }
            factor[row * d + column] = coefficient * scales[row];
        }
        remaining.retain(|&row| row != pivot);
    }
    Ok(OrthogonalRows {
        factor,
        basis,
        indices,
    })
}

/// Pivoted LDL factorization with power-of-two equilibration and compensated
/// Schur complements. Exact low-rank process matrices must not acquire spurious
/// sqrt(epsilon) innovations from Cholesky's rounded square roots.
fn process_root(matrix: &[f64], d: usize) -> Result<Vec<f64>, ()> {
    let scales: Vec<f64> = (0..d)
        .map(|i| {
            let root = matrix[i * d + i].sqrt();
            if root > 0.0 {
                root.log2().floor().exp2()
            } else {
                1.0
            }
        })
        .collect();
    let mut residual = matrix.to_vec();
    for i in 0..d {
        for j in 0..d {
            residual[i * d + j] = residual[i * d + j] / scales[i] / scales[j];
        }
    }
    let mut remaining: Vec<usize> = (0..d).collect();
    let mut factor = vec![0.0; d * d];
    let tolerance = 256.0 * f64::EPSILON * d as f64;
    for column in 0..d {
        let pivot = *remaining
            .iter()
            .max_by(|&&a, &&b| residual[a * d + a].total_cmp(&residual[b * d + b]))
            .ok_or(())?;
        let variance = residual[pivot * d + pivot];
        if variance <= 0.0 {
            if remaining.iter().any(|&i| {
                remaining
                    .iter()
                    .any(|&j| residual[i * d + j].abs() > tolerance)
            }) {
                return Err(());
            }
            break;
        }
        for &i in &remaining {
            factor[i * d + column] = residual[i * d + pivot] / variance.sqrt() * scales[i];
        }
        remaining.retain(|&i| i != pivot);
        for &i in &remaining {
            for &j in &remaining {
                let product = residual[i * d + pivot] * residual[j * d + pivot];
                let error = (-residual[i * d + pivot]).mul_add(residual[j * d + pivot], product);
                residual[i * d + j] =
                    (residual[i * d + j].mul_add(variance, -product) + error) / variance;
            }
        }
    }
    Ok(factor)
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
            // Compare in covariance units for this pair of states. An absolute
            // floor would accept gross asymmetry when states have small units.
            let marginal_scale = matrix[i * d + i].abs().sqrt() * matrix[j * d + j].abs().sqrt();
            let scale = marginal_scale.max(a.abs()).max(b.abs());
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

/// Factor a covariance after diagonal equilibration, so numerical rank is
/// measured in correlation units rather than the units of the largest state.
/// Complete diagonal pivoting keeps roundoff in rank-deficient FFBS matrices
/// from being amplified by a nearly zero pivot. The returned factor is dense:
/// its rows are in the original state order and A = factor * factor'.
fn positive_semidefinite_factor(matrix: &[f64], d: usize) -> Result<Vec<f64>, ()> {
    let tolerance = 64.0 * f64::EPSILON * d as f64;
    let mut scales = vec![0.0; d];
    for i in 0..d {
        let variance = matrix[i * d + i];
        if !variance.is_finite() || variance < 0.0 {
            return Err(());
        }
        scales[i] = variance.sqrt();
    }
    let mut residual = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let value = matrix[i * d + j];
            if !value.is_finite() {
                return Err(());
            }
            let correlation = if scales[i] == 0.0 || scales[j] == 0.0 {
                // A PSD matrix with zero marginal variance has a zero row.
                if value != 0.0 {
                    return Err(());
                }
                0.0
            } else if i == j {
                1.0
            } else {
                // Divide by the larger scale first to avoid overflow even
                // when the two state variances have very different units.
                value / scales[i].max(scales[j]) / scales[i].min(scales[j])
            };
            if !correlation.is_finite() || correlation.abs() > 1.0 + tolerance {
                return Err(());
            }
            residual[i * d + j] = correlation;
            residual[j * d + i] = correlation;
        }
    }
    // Preserve every positive pivot when ordinary Cholesky succeeds. The
    // rank-revealing fallback is needed only for semidefinite matrices (and
    // their roundoff perturbations), not for small positive variances.
    if let Ok(mut factor) = cholesky(&residual, d) {
        for i in 0..d {
            for j in 0..=i {
                factor[i * d + j] *= scales[i];
            }
        }
        return Ok(factor);
    }
    let mut remaining: Vec<usize> = (0..d).collect();
    let mut factor = vec![0.0; d * d];
    for column in 0..d {
        let pivot_position = (column..d)
            .max_by(|&a, &b| {
                residual[remaining[a] * d + remaining[a]]
                    .total_cmp(&residual[remaining[b] * d + remaining[b]])
            })
            .ok_or(())?;
        remaining.swap(column, pivot_position);
        let pivot = remaining[column];
        let variance = residual[pivot * d + pivot];
        if variance <= tolerance {
            // Only a residual entirely within roundoff can be discarded;
            // negative diagonals or off-diagonals can otherwise hide an
            // indefinite matrix behind an apparently zero pivot.
            for &i in &remaining[column..] {
                for &j in &remaining[column..] {
                    if residual[i * d + j].abs() > tolerance {
                        return Err(());
                    }
                }
            }
            break;
        }
        let root = variance.sqrt();
        for &i in &remaining[column..] {
            factor[i * d + column] = residual[i * d + pivot] / root;
        }
        for &i in &remaining[column + 1..] {
            for &j in &remaining[column + 1..] {
                residual[i * d + j] -= factor[i * d + column] * factor[j * d + column];
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            factor[i * d + j] *= scales[i];
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
        for column in 0..d {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn singular_transition_smoothing_matches_scalar_gaussian_conditioning() {
        // x[t] = (0.5*z[t-1], z[t-1]) loses the second initial coordinate.
        // The remaining trajectory is a one-parameter Gaussian regression.
        let model = LinearGaussianStateSpace::new(
            2,
            vec![0.5, 0.0, 1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0; 4],
            1.0,
            vec![0.0; 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let y = [0.2, f64::NAN, -0.1, 0.3];
        let h: Vec<f64> = (1..=4).map(|t| 0.5_f64.powi(t)).collect();
        let variance = 1.0
            / (1.0
                + h.iter()
                    .zip(y)
                    .filter(|(_, y)| y.is_finite())
                    .map(|(h, _)| h * h)
                    .sum::<f64>());
        let mean = variance
            * h.iter()
                .zip(y)
                .filter(|(_, y)| y.is_finite())
                .map(|(h, y)| h * y)
                .sum::<f64>();
        let smooth = model.smooth(&y).unwrap();
        for (time, &h) in h.iter().enumerate() {
            for (i, coefficient) in [h, 2.0 * h].into_iter().enumerate() {
                assert_close(smooth.smoothed_means[time][i], coefficient * mean);
                for (j, other) in [h, 2.0 * h].into_iter().enumerate() {
                    assert_close(
                        smooth.smoothed_covariances[time][i * 2 + j],
                        coefficient * other * variance,
                    );
                }
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(751);
        let mut sum = [0.0; 2];
        let mut squares = [0.0; 2];
        for _ in 0..8000 {
            let states = model.sample_states_ffbs(&y, &mut rng).unwrap();
            for (i, value) in states[0].iter().enumerate() {
                sum[i] += value;
                squares[i] += value * value;
            }
            for pair in states.windows(2) {
                assert!((pair[1][0] - 0.5 * pair[0][0]).abs() < 1e-7);
                assert!((pair[1][1] - pair[0][0]).abs() < 1e-7);
            }
        }
        for (i, expected_mean, expected_variance) in [(0, mean, variance), (1, 0.0, 1.0)] {
            let actual_mean = sum[i] / 8000.0;
            assert!((actual_mean - expected_mean).abs() < 0.035);
            assert!((squares[i] / 8000.0 - actual_mean.powi(2) - expected_variance).abs() < 0.04);
        }
    }

    #[test]
    fn mixed_rank_collapse_matches_independent_gaussian_updates() {
        // T^2 = T Q = 0. The first state has two random directions; all
        // subsequent states have one and are mutually independent.
        let w = [1.0, -1.0, 2.0];
        let q: Vec<f64> = w
            .iter()
            .flat_map(|a| w.iter().map(move |b| a * b / 8.0))
            .collect();
        let model = LinearGaussianStateSpace::new(
            3,
            vec![0.25, -0.25, -0.25, 0.25, -0.25, -0.25, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            q.clone(),
            1.0,
            vec![0.0; 3],
            identity(3),
        )
        .unwrap();
        let y = [0.2, -0.4, 0.7];
        let result = model.smooth(&y).unwrap();
        for (time, observation) in y.iter().enumerate() {
            let mut prior = q.clone();
            if time == 0 {
                for i in 0..2 {
                    for j in 0..2 {
                        prior[i * 3 + j] += 3.0 / 16.0;
                    }
                }
            }
            let variance = 1.0 + prior[0];
            for i in 0..3 {
                assert_close(
                    result.smoothed_means[time][i],
                    prior[i * 3] / variance * observation,
                );
                for j in 0..3 {
                    assert_close(
                        result.smoothed_covariances[time][i * 3 + j],
                        prior[i * 3 + j] - prior[i * 3] * prior[j * 3] / variance,
                    );
                }
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(872);
        let mut sum = 0.0;
        let mut square = 0.0;
        for _ in 0..8000 {
            let states = model.sample_states_ffbs(&y, &mut rng).unwrap();
            for state in &states[2..] {
                assert!((state[1] + state[0]).abs() < 1e-12);
                assert!((state[2] - 2.0 * state[0]).abs() < 1e-12);
            }
            sum += states[3][0];
            square += states[3][0].powi(2);
        }
        let mean = sum / 8000.0;
        assert!((mean - 0.7 / 9.0).abs() < 0.012);
        assert!((square / 8000.0 - mean * mean - 1.0 / 9.0).abs() < 0.008);
    }

    #[test]
    fn roots_preserve_small_positive_modes_and_reject_nonfinite_draws() {
        let covariance = vec![1.0, 1.0 - f64::EPSILON, 1.0 - f64::EPSILON, 1.0];
        let root = process_root(&covariance, 2).unwrap();
        assert!(root[1].abs() + root[3].abs() > 0.0);
        let model = LinearGaussianStateSpace::new(
            2,
            identity(2),
            vec![1.0, -1.0],
            vec![0.0; 4],
            1e-14,
            vec![0.0; 2],
            covariance,
        )
        .unwrap();
        let (conditionals, _) = model.backward_parameters(&[1e-8, -1e-8, 2e-8]).unwrap();
        for conditional in conditionals {
            assert_eq!(conditional.gain, identity(2));
            assert!(conditional.factor.iter().all(|x| *x == 0.0));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(871);
        assert!(sample_from_factor(&[f64::INFINITY], &[0.0], 1, 1, &mut rng).is_err());
    }

    #[test]
    fn zero_transition_preserves_initial_uncertainty_and_exact_future_state() {
        let model = LinearGaussianStateSpace::new(
            1,
            vec![0.0],
            vec![1.0],
            vec![0.0],
            1.0,
            vec![3.0],
            vec![2.0],
        )
        .unwrap();
        let y = [1.0, -2.0, 0.5];
        let smooth = model.smooth(&y).unwrap();
        assert_eq!(smooth.smoothed_means, vec![vec![0.0]; 3]);
        assert_eq!(smooth.smoothed_covariances, vec![vec![0.0]; 3]);
        let mut rng = ChaCha8Rng::seed_from_u64(754);
        let mut sum = 0.0;
        let mut squares = 0.0;
        for _ in 0..5000 {
            let states = model.sample_states_ffbs(&y, &mut rng).unwrap();
            assert_eq!(states[1..], vec![vec![0.0]; 3]);
            sum += states[0][0];
            squares += states[0][0].powi(2);
        }
        let mean = sum / 5000.0;
        assert!((mean - 3.0).abs() < 0.07);
        assert!((squares / 5000.0 - mean * mean - 2.0).abs() < 0.1);
        assert!(process_root(&[1.0, 2.0, 2.0, 1.0], 2).is_err());
    }

    #[test]
    fn smoother_preserves_small_posterior_variance_after_diffuse_missing_state() {
        let model = LinearGaussianStateSpace::local_level(0.0, 1.0, 0.0, 1e20).unwrap();
        let result = model.smooth(&[f64::NAN, 0.0]).unwrap();
        for covariance in result.smoothed_covariances {
            // Static latent state: both marginals have precision 1e-20 + 1.
            assert_close(covariance[0], 1.0);
        }
    }

    #[test]
    fn covariance_factor_preserves_small_and_heteroscaled_correlations() {
        let correlation = 1.0 - f64::EPSILON;
        let factor =
            positive_semidefinite_factor(&[1.0, correlation, correlation, 1.0], 2).unwrap();
        assert!(
            factor[3] > 0.0,
            "a strictly positive pivot must be retained"
        );
        for covariance in [
            vec![1e-14, 5e-15, 5e-15, 1e-14],
            vec![1e-14, 5e-8, 5e-8, 1.0],
            vec![1e-300, 0.5, 0.5, 1e300],
            vec![1e-14, 1e-7, 1e-7, 1.0], // rank one
        ] {
            let factor = positive_semidefinite_factor(&covariance, 2).unwrap();
            let reconstructed = mat_mul_transpose_right(&factor, &factor, 2);
            for (actual, expected) in reconstructed.iter().zip(&covariance) {
                assert!(
                    (actual / expected - 1.0).abs() < 1e-13,
                    "{actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn covariance_symmetry_validation_is_invariant_to_state_units() {
        for scale in [1e-20, 1.0, 1e20] {
            let covariance = vec![scale, 0.0, 0.5 * scale, scale];
            assert!(matches!(
                LinearGaussianStateSpace::new(
                    2,
                    identity(2),
                    vec![1.0; 2],
                    covariance,
                    1.0,
                    vec![0.0; 2],
                    identity(2),
                ),
                Err(StateSpaceError::NotSymmetric(_))
            ));
        }
    }

    #[test]
    fn covariance_validation_rejects_indefinite_matrices_at_any_scale() {
        for scale in [1e-20, 1.0, 1e20] {
            for covariance in [
                vec![-scale, 0.0, 0.0, scale],
                vec![scale, 2.0 * scale, 2.0 * scale, scale],
                vec![0.0, 1e-8 * scale, 1e-8 * scale, scale],
            ] {
                assert!(positive_semidefinite_factor(&covariance, 2).is_err());
            }
        }
        // A zero remaining diagonal must not conceal an indefinite residual.
        assert!(
            positive_semidefinite_factor(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.0, 1.01, 1.0], 3)
                .is_err()
        );
    }

    #[test]
    fn initial_and_process_draws_preserve_small_covariance_forecast_variance() {
        let covariance = vec![1e-14, 5e-15, 5e-15, 1e-14];
        let model = LinearGaussianStateSpace::new(
            2,
            identity(2),
            vec![1e7, 1e7],
            covariance.clone(),
            1.0,
            vec![0.0; 2],
            covariance,
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(408);
        let draws = 20000;
        let mut second_moments = [0.0; 2];
        let mut observed_second_moment = 0.0;
        for _ in 0..draws {
            let initial = model.simulate_initial(&mut rng).unwrap();
            let process = model.simulate_transition(&[0.0; 2], &mut rng).unwrap();
            for (index, state) in [initial, process].iter().enumerate() {
                let mean = dot(&model.observation, state);
                second_moments[index] += mean * mean;
                if index == 0 {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    observed_second_moment += (mean + z).powi(2);
                }
            }
        }
        for sum in second_moments {
            assert!((sum / draws as f64 - 3.0).abs() < 0.1);
        }
        assert!((observed_second_moment / draws as f64 - 4.0).abs() < 0.13);
    }

    #[test]
    fn filtering_and_ffbs_are_invariant_to_state_and_predictor_units() {
        let observations = [0.3, -0.2, f64::NAN, 0.5];
        let make_model = |scales: [f64; 2]| {
            let mut covariance = vec![1.0, 0.5, 0.5, 1.0];
            for i in 0..2 {
                for j in 0..2 {
                    covariance[i * 2 + j] *= scales[i] * scales[j];
                }
            }
            LinearGaussianStateSpace::new(
                2,
                identity(2),
                vec![1.0 / scales[0], 1.0 / scales[1]],
                vec![0.0; 4],
                1.0,
                vec![0.0; 2],
                covariance,
            )
            .unwrap()
        };
        let reference = make_model([1.0, 1.0]).smooth(&observations).unwrap();
        for scales in [[1e-7, 1e-7], [1e-7, 1.0], [1e7, 1e-7]] {
            let model = make_model(scales);
            let smoother = model.smooth(&observations).unwrap();
            assert_close(
                smoother.filter.log_likelihood,
                reference.filter.log_likelihood,
            );
            for t in 0..observations.len() {
                for i in 0..2 {
                    assert_close(
                        smoother.smoothed_means[t][i] / scales[i],
                        reference.smoothed_means[t][i],
                    );
                    for j in 0..2 {
                        assert_close(
                            smoother.smoothed_covariances[t][i * 2 + j] / scales[i] / scales[j],
                            reference.smoothed_covariances[t][i * 2 + j],
                        );
                    }
                }
            }
            let expected = reference.smoothed_covariances.last().unwrap();
            let mean = reference.smoothed_means.last().unwrap();
            let mut sums = [0.0; 4];
            let mut rng = ChaCha8Rng::seed_from_u64(35);
            let draws = 6000;
            for _ in 0..draws {
                let states = model.sample_states_ffbs(&observations, &mut rng).unwrap();
                for pair in states.windows(2) {
                    for (i, scale) in scales.iter().enumerate() {
                        assert!(((pair[0][i] - pair[1][i]) / scale).abs() < 1e-8);
                    }
                }
                let last = states.last().unwrap();
                for i in 0..2 {
                    for j in 0..2 {
                        sums[i * 2 + j] +=
                            (last[i] / scales[i] - mean[i]) * (last[j] / scales[j] - mean[j]);
                    }
                }
            }
            for (sum, expected) in sums.iter().zip(expected) {
                assert!((sum / draws as f64 - expected).abs() < 0.025);
            }
        }
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
    fn varying_observation_noise_matches_precision_weighted_static_posterior() {
        let model = LinearGaussianStateSpace::local_level(0.0, 1.0, 0.0, 2.0)
            .unwrap()
            .with_observation_variances(vec![0.5, 2.0])
            .unwrap();
        let filtered = model.filter(&[1.0, 3.0]).unwrap();
        // precision = 1/2 + 1/.5 + 1/2 = 3; precision-weighted data = 3.5.
        assert_close(filtered.filtered_means[1][0], 3.5 / 3.0);
        assert_close(filtered.filtered_covariances[1][0], 1.0 / 3.0);
        let augmented = model
            .with_static_regression(&[vec![0.0], vec![0.0]], &[0.0], &[1.0])
            .unwrap();
        assert_close(
            augmented.filter(&[1.0, 3.0]).unwrap().filtered_means[1][0],
            3.5 / 3.0,
        );
        assert!(model.filter(&[1.0]).is_err());
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
