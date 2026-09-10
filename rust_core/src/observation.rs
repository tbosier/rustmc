//! Shared observation simulation and conditional means for graph models.
use crate::graph::ObsFamily;
use rand::Rng;
use rand_distr::{Distribution, Exp, Gamma, Normal, Poisson};

fn positive(value: f64, label: &str) -> Result<f64, String> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("{label} must be finite and positive; got {value}"))
    }
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
/// Expected response, on the observation scale.
pub fn mean(family: ObsFamily, eta: f64, aux: Option<f64>) -> Result<f64, String> {
    let value = match family {
        ObsFamily::Normal => eta,
        ObsFamily::BernoulliLogit => sigmoid(eta),
        ObsFamily::PoissonLog | ObsFamily::NegativeBinomialLog => eta.exp(),
        ObsFamily::ExponentialLog => (-eta).exp(),
        ObsFamily::LogNormal => {
            let s = positive(aux.ok_or("missing sigma")?, "sigma")?;
            (eta + 0.5 * s * s).exp()
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err("observation mean is not representable".into())
    }
}
/// Simulate the specified family without clipping valid rates or outcomes.
pub fn sample<R: Rng + ?Sized>(
    family: ObsFamily,
    eta: f64,
    aux: Option<f64>,
    rng: &mut R,
) -> Result<f64, String> {
    if !eta.is_finite() {
        return Err("linear predictor must be finite".into());
    }
    let value = match family {
        ObsFamily::Normal | ObsFamily::LogNormal => {
            let sigma = positive(aux.ok_or("missing sigma")?, "sigma")?;
            let x = Normal::new(eta, sigma)
                .map_err(|e| e.to_string())?
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
        ObsFamily::PoissonLog => Poisson::new(positive(eta.exp(), "Poisson rate")?)
            .map_err(|e| e.to_string())?
            .sample(rng),
        ObsFamily::ExponentialLog => Exp::new(positive(eta.exp(), "Exponential rate")?)
            .map_err(|e| e.to_string())?
            .sample(rng),
        ObsFamily::NegativeBinomialLog => {
            let alpha = positive(aux.ok_or("missing alpha")?, "alpha")?;
            let scale = positive(eta.exp() / alpha, "Gamma scale")?;
            let lambda = Gamma::new(alpha, scale)
                .map_err(|e| e.to_string())?
                .sample(rng);
            if lambda == 0.0 {
                0.0
            } else {
                Poisson::new(positive(lambda, "Poisson rate")?)
                    .map_err(|e| e.to_string())?
                    .sample(rng)
            }
        }
    };
    if !value.is_finite() || (family == ObsFamily::LogNormal && value == 0.0) {
        Err("simulated observation is not representable".into())
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
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
