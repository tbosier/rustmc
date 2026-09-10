//! Elliptical slice sampling for a standard-normal latent block.
//!
//! The callback is the log likelihood (or all non-Gaussian factors), excluding
//! the standard-normal prior. Other coordinates are held fixed. Bracket
//! exhaustion returns an error; it never silently discards a transition.
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn update<R: Rng + ?Sized>(
    state: &mut [f64],
    block: std::ops::Range<usize>,
    current_log_likelihood: f64,
    log_likelihood: impl Fn(&[f64]) -> f64,
    rng: &mut R,
) -> Result<(f64, usize), String> {
    if block.is_empty()
        || block.end > state.len()
        || !current_log_likelihood.is_finite()
        || state.iter().any(|v| !v.is_finite())
    {
        return Err(
            "ESS needs finite states, a nonempty valid block and finite current log likelihood"
                .into(),
        );
    }
    let direction: Vec<f64> = block.clone().map(|_| StandardNormal.sample(rng)).collect();
    let original = state[block.clone()].to_vec();
    let threshold = current_log_likelihood + (1.0 - rng.gen::<f64>()).ln();
    let mut angle = rng.gen::<f64>() * std::f64::consts::TAU;
    let mut lower = angle - std::f64::consts::TAU;
    let mut upper = angle;
    for evaluations in 1..=100_000 {
        for (j, i) in block.clone().enumerate() {
            state[i] = original[j] * angle.cos() + direction[j] * angle.sin();
        }
        let candidate = log_likelihood(state);
        if candidate.is_nan() || candidate == f64::INFINITY {
            state[block].copy_from_slice(&original);
            return Err("ESS likelihood returned NaN or positive infinity".into());
        }
        if candidate >= threshold {
            return Ok((candidate, evaluations));
        }
        if angle < 0.0 {
            lower = angle;
        } else {
            upper = angle;
        }
        angle = lower + rng.gen::<f64>() * (upper - lower);
    }
    state[block].copy_from_slice(&original);
    Err("ESS bracket exhausted; no sample was returned".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn rejects_nonfinite_states_even_with_constant_likelihood() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12);
        let mut state = [f64::INFINITY];
        assert!(update(&mut state, 0..1, 0.0, |_| 0.0, &mut rng).is_err());
        assert_eq!(state[0], f64::INFINITY);
    }
    #[test]
    fn normal_likelihood_matches_analytic_posterior() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(702);
        let likelihood = |x: &[f64]| -0.5 * (x[0] - 2.0).powi(2) / 0.25;
        let mut x = [0.0];
        let mut lp = likelihood(&x);
        let mut samples = Vec::new();
        for i in 0..22000 {
            lp = update(&mut x, 0..1, lp, likelihood, &mut rng).unwrap().0;
            if i >= 2000 {
                samples.push(x[0]);
            }
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!((mean - 1.6).abs() < 0.02, "{mean}");
        assert!((variance - 0.2).abs() < 0.015, "{variance}");
    }
}
