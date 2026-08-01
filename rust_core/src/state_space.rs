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

const SYMMETRY_TOLERANCE: f64 = 1e-10;
const LOG_2_PI: f64 = 1.8378770664093453;

#[derive(Debug, Clone, PartialEq)]
pub enum StateSpaceError {
    InvalidDimension(String),
    InvalidParameter(String),
    NonFinite(String),
    NotSymmetric(String),
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
        cholesky(&process_covariance, dimension).map_err(|_| {
            StateSpaceError::NotPositiveDefinite(
                "process covariance must be strictly positive definite".into(),
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
        Ok(ForecastResult {
            state_means,
            state_covariances,
            observation_means,
            observation_variances,
        })
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
        assert!(matches!(
            LinearGaussianStateSpace::local_level(0.0, 1.0, 0.0, 1.0),
            Err(StateSpaceError::NotPositiveDefinite(_))
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
