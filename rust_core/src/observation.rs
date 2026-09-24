//! Shared observation simulation and conditional means for graph models.
use crate::graph::ObsFamily;
use rand::Rng;
use rand_distr::{Distribution, Exp, Gamma, Normal};

/// Why an observation density, mean or draw is unavailable. The message is
/// the whole explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationError {
    /// An observation, linear predictor or auxiliary parameter is missing or
    /// outside the family's support.
    InvalidInput(String),
    /// The result, or a draw on the way to it, is not representable.
    Unrepresentable(String),
}

impl std::fmt::Display for ObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Unrepresentable(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ObservationError {}

/// Model-level callers carry these as messages in their own error types.
impl From<ObservationError> for String {
    fn from(error: ObservationError) -> Self {
        error.to_string()
    }
}

impl From<crate::count_sampling::CountSamplingError> for ObservationError {
    fn from(error: crate::count_sampling::CountSamplingError) -> Self {
        Self::Unrepresentable(error.to_string())
    }
}

fn invalid(message: impl Into<String>) -> ObservationError {
    ObservationError::InvalidInput(message.into())
}

fn unrepresentable(message: impl Into<String>) -> ObservationError {
    ObservationError::Unrepresentable(message.into())
}

fn required(aux: Option<f64>, label: &str) -> Result<f64, ObservationError> {
    aux.ok_or_else(|| invalid(format!("missing {label}")))
}

fn positive(value: f64, label: &str) -> Result<f64, ObservationError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "{label} must be finite and positive; got {value}"
        )))
    }
}
/// Logistic function, evaluated so that neither tail overflows.
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Pointwise log likelihood on the same parameter scales as observation sampling.
/// Invalid observations/parameters are errors; an unrepresentably small density
/// may legitimately have log probability negative infinity.
pub fn log_density(
    family: ObsFamily,
    observed: f64,
    eta: f64,
    aux: Option<f64>,
) -> Result<f64, ObservationError> {
    if !observed.is_finite() || !eta.is_finite() {
        return Err(invalid("observation and linear predictor must be finite"));
    }
    let logp = match family {
        ObsFamily::Normal | ObsFamily::LogNormal => {
            let sigma = positive(required(aux, "sigma")?, "sigma")?;
            let (response, jacobian) = if family == ObsFamily::LogNormal {
                let response = positive(observed, "LogNormal observation")?.ln();
                (response, response)
            } else {
                (observed, 0.0)
            };
            let z = (response - eta) / sigma;
            -0.5 * std::f64::consts::TAU.ln() - sigma.ln() - jacobian - 0.5 * z * z
        }
        ObsFamily::BernoulliLogit => {
            if observed != 0.0 && observed != 1.0 {
                return Err(invalid("Bernoulli observation must be zero or one"));
            }
            if observed == 1.0 {
                -crate::autodiff::softplus(-eta)
            } else {
                -crate::autodiff::softplus(eta)
            }
        }
        ObsFamily::PoissonLog | ObsFamily::NegativeBinomialLog => {
            if observed < 0.0 || observed.fract() != 0.0 {
                return Err(invalid("count observation must be a nonnegative integer"));
            }
            if family == ObsFamily::PoissonLog {
                crate::count_sampling::log_mass_from_log_rate(observed, eta)
            } else {
                let alpha = positive(required(aux, "alpha")?, "alpha")?;
                crate::negative_binomial::log_mass(observed, eta, alpha)
            }
        }
        ObsFamily::ExponentialLog => {
            if observed < 0.0 {
                return Err(invalid("Exponential observation must be nonnegative"));
            }
            eta - observed * eta.exp()
        }
    };
    if logp.is_nan() {
        Err(unrepresentable(
            "observation log likelihood is not representable",
        ))
    } else {
        Ok(logp)
    }
}
/// Expected response, on the observation scale.
pub fn mean(family: ObsFamily, eta: f64, aux: Option<f64>) -> Result<f64, ObservationError> {
    let value = match family {
        ObsFamily::Normal => eta,
        ObsFamily::BernoulliLogit => sigmoid(eta),
        ObsFamily::PoissonLog | ObsFamily::NegativeBinomialLog => eta.exp(),
        ObsFamily::ExponentialLog => (-eta).exp(),
        ObsFamily::LogNormal => {
            let s = positive(required(aux, "sigma")?, "sigma")?;
            (eta + 0.5 * s * s).exp()
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(unrepresentable("observation mean is not representable"))
    }
}
/// Simulate the specified family without clipping valid rates or outcomes.
pub fn sample<R: Rng + ?Sized>(
    family: ObsFamily,
    eta: f64,
    aux: Option<f64>,
    rng: &mut R,
) -> Result<f64, ObservationError> {
    if !eta.is_finite() {
        return Err(invalid("linear predictor must be finite"));
    }
    let value = match family {
        ObsFamily::Normal | ObsFamily::LogNormal => {
            let sigma = positive(required(aux, "sigma")?, "sigma")?;
            let x = Normal::new(eta, sigma)
                .map_err(|e| invalid(e.to_string()))?
                .sample(rng);
            if family == ObsFamily::LogNormal {
                x.exp()
            } else {
                x
            }
        }
        ObsFamily::BernoulliLogit => {
            if rng.gen::<f64>() < sigmoid(eta) {
                1.0
            } else {
                0.0
            }
        }
        ObsFamily::PoissonLog => {
            crate::count_sampling::poisson(positive(eta.exp(), "Poisson rate")?, rng)?
        }
        ObsFamily::ExponentialLog => Exp::new(positive(eta.exp(), "Exponential rate")?)
            .map_err(|e| invalid(e.to_string()))?
            .sample(rng),
        ObsFamily::NegativeBinomialLog => {
            let alpha = positive(required(aux, "alpha")?, "alpha")?;
            let scale = positive(eta.exp() / alpha, "Gamma scale")?;
            let lambda = Gamma::new(alpha, scale)
                .map_err(|e| invalid(e.to_string()))?
                .sample(rng);
            if lambda == 0.0 {
                0.0
            } else {
                crate::count_sampling::poisson(positive(lambda, "Poisson rate")?, rng)?
            }
        }
    };
    if !value.is_finite() || (family == ObsFamily::LogNormal && value == 0.0) {
        Err(unrepresentable(
            "simulated observation is not representable",
        ))
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn small_poisson_and_negative_binomial_means_never_produce_negative_counts() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(91);
        for family in [ObsFamily::PoissonLog, ObsFamily::NegativeBinomialLog] {
            for _ in 0..1000 {
                let draw = sample(family, -50.0, Some(5.0), &mut rng).unwrap();
                assert!(draw >= 0.0 && draw.fract() == 0.0);
            }
        }
    }

    #[test]
    fn errors_say_whether_the_input_or_the_result_was_at_fault() {
        assert_eq!(
            log_density(ObsFamily::BernoulliLogit, 2.0, 0.0, None),
            Err(ObservationError::InvalidInput(
                "Bernoulli observation must be zero or one".into()
            ))
        );
        assert_eq!(
            mean(ObsFamily::LogNormal, 0.0, None),
            Err(ObservationError::InvalidInput("missing sigma".into()))
        );
        assert_eq!(
            mean(ObsFamily::PoissonLog, 1000.0, None),
            Err(ObservationError::Unrepresentable(
                "observation mean is not representable".into()
            ))
        );
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let error = sample(ObsFamily::PoissonLog, 37.0, None, &mut rng).unwrap_err();
        assert_eq!(
            error,
            ObservationError::Unrepresentable(
                "Poisson rate outside the supported exact-count range".into()
            )
        );
        assert_eq!(String::from(error.clone()), error.to_string());
    }

    #[test]
    fn exponential_extreme_scales_preserve_moments() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(56);
        for eta in [-40.0_f64, 40.0] {
            let values: Vec<_> = (0..100_000)
                .map(|_| sample(ObsFamily::ExponentialLog, eta, None, &mut rng).unwrap())
                .collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            assert!((mean / (-eta).exp() - 1.0).abs() < 0.02);
            assert!(values.iter().all(|x| *x >= 0.0));
        }
    }
}

#[cfg(test)]
mod family_moment_tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn family_support_and_conditional_moments() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(801);
        for (family, eta, aux) in [
            (ObsFamily::Normal, 1.5, Some(0.7)),
            (ObsFamily::BernoulliLogit, 0.3, None),
            (ObsFamily::PoissonLog, 0.8, None),
            (ObsFamily::LogNormal, 0.2, Some(0.4)),
            (ObsFamily::NegativeBinomialLog, 0.7, Some(2.0)),
        ] {
            let values: Vec<_> = (0..50_000)
                .map(|_| sample(family, eta, aux, &mut rng).unwrap())
                .collect();
            let observed = values.iter().sum::<f64>() / values.len() as f64;
            assert!((observed / mean(family, eta, aux).unwrap() - 1.0).abs() < 0.035);
            if matches!(
                family,
                ObsFamily::BernoulliLogit | ObsFamily::PoissonLog | ObsFamily::NegativeBinomialLog
            ) {
                assert!(values.iter().all(|v| *v >= 0.0 && v.fract() == 0.0));
            }
            if family == ObsFamily::LogNormal {
                assert!(values.iter().all(|v| *v > 0.0));
            }
        }
    }
}
