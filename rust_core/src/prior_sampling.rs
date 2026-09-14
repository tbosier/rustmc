//! Analytic prior draws in unconstrained coordinates, without arbitrary floors.
use rand::{distributions::Open01, Rng};
use rand_distr::{Distribution, Gamma, StandardNormal};

fn positive(value: f64) -> Result<f64, String> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("prior parameters must be finite and positive".into())
    }
}

fn finite(value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("unconstrained prior draw is not representable".into())
    }
}

/// Log of a Gamma(shape, rate) draw. For shape below one, the identity
/// G(shape) = G(shape + 1) U^(1/shape) avoids underflow before taking the log.
pub fn log_gamma<R: Rng + ?Sized>(shape: f64, rate: f64, rng: &mut R) -> Result<f64, String> {
    positive(shape)?;
    positive(rate)?;
    let boosted = if shape < 1.0 { shape + 1.0 } else { shape };
    let unit: f64 = Gamma::new(boosted, 1.0)
        .map_err(|e| e.to_string())?
        .sample(rng);
    let mut draw = positive(unit)?.ln() - rate.ln();
    if shape < 1.0 {
        let u: f64 = rng.sample(Open01);
        draw += u.ln() / shape;
    }
    finite(draw)
}

/// The logit of a Beta draw is the difference of independent log Gamma draws.
pub fn logit_beta<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64, String> {
    finite(log_gamma(alpha, 1.0, rng)? - log_gamma(beta, 1.0, rng)?)
}

pub fn log_half_normal<R: Rng + ?Sized>(sigma: f64, rng: &mut R) -> Result<f64, String> {
    positive(sigma)?;
    let standard: f64 = StandardNormal.sample(rng);
    finite(positive(standard.abs())?.ln() + sigma.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn small_gamma_shapes_retain_unconstrained_tail_mass() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(951);
        let draws = 50_000;
        let count = (0..draws)
            .filter(|_| log_gamma(0.001, 1.0, &mut rng).unwrap() < -1000.0)
            .count();
        // Gamma CDF below exp(-1000) is exp(-1)/Gamma(1.001), to negligible error.
        let expected = (-1.0 - crate::autodiff::ln_gamma(1.001)).exp();
        assert!((count as f64 / draws as f64 - expected).abs() < 0.012);
    }

    #[test]
    fn beta_draws_retain_both_logit_tails() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(952);
        let values: Vec<_> = (0..50_000)
            .map(|_| logit_beta(0.01, 0.01, &mut rng).unwrap())
            .collect();
        for sign in [-1.0, 1.0] {
            let fraction =
                values.iter().filter(|&&x| sign * x > 40.0).count() as f64 / values.len() as f64;
            assert!((fraction - 0.3352143658).abs() < 0.012);
        }
    }
}
