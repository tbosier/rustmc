//! A Bernoulli hurdle with dynamic lognormal positive amounts.
//!
//! `P(y[t] > 0) = p`, `p ~ Beta(a,b)`, and conditional on a positive amount,
//! `log(y[t]) = level[t] + Normal(0,r)`, with a Gaussian random-walk level
//! and explicit inverse-gamma priors on q and r. Zeros inform occurrence only;
//! missing values inform neither component, but both retain their time position.
//! Occurrence is static and independent of severity a priori. This factorization
//! permits exact Beta occurrence draws and conjugate Gaussian severity FFBS/Gibbs.

use crate::bayesian_forecast::{
    BayesianForecastError, InverseGammaPrior, PosteriorPredictiveForecast,
};
use crate::state_space::LinearGaussianStateSpace;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

const MAX_VALUES: usize = 25_000_000;

#[derive(Debug, Clone)]
pub struct HurdleLogNormalConfig {
    pub occurrence_alpha: f64,
    pub occurrence_beta: f64,
    pub initial_log_level: f64,
    pub initial_variance: f64,
    pub process_variance_prior: InverseGammaPrior,
    pub observation_variance_prior: InverseGammaPrior,
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub thinning: usize,
    pub seed: u64,
}

impl HurdleLogNormalConfig {
    pub fn validate(&self) -> Result<(), BayesianForecastError> {
        for (name, value) in [
            ("occurrence alpha", self.occurrence_alpha),
            ("occurrence beta", self.occurrence_beta),
            ("initial variance", self.initial_variance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!("{name} must be finite and positive")));
            }
        }
        if !self.initial_log_level.is_finite() {
            return Err(invalid("initial log level must be finite"));
        }
        InverseGammaPrior::new(
            self.process_variance_prior.shape,
            self.process_variance_prior.scale,
        )?;
        InverseGammaPrior::new(
            self.observation_variance_prior.shape,
            self.observation_variance_prior.scale,
        )?;
        if self.num_chains == 0 || self.num_draws == 0 || self.thinning == 0 {
            return Err(invalid("chains, draws, and thinning must be positive"));
        }
        self.num_draws
            .checked_mul(self.thinning)
            .and_then(|n| n.checked_add(self.num_warmup))
            .ok_or_else(|| invalid("iteration count overflow"))?;
        allocation(&[self.num_chains, self.num_draws, 4])?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HurdleLogNormalDraw {
    pub payment_probability: f64,
    pub process_variance: f64,
    pub observation_variance: f64,
    pub terminal_log_level: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HurdleLogNormalPosterior {
    /// Joint samples indexed `[chain][draw]`.
    pub chains: Vec<Vec<HurdleLogNormalDraw>>,
    pub time_count: usize,
    pub observed_count: usize,
    pub positive_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HurdleLogNormalForecast {
    /// `state_paths` are E[y | p, future log level, observation variance].
    /// `observation_paths` include exact zeros and positive payment realizations.
    pub paths: PosteriorPredictiveForecast,
    /// E[y | y > 0, future log level, observation variance].
    pub positive_mean_paths: Vec<Vec<Vec<f64>>>,
}

/// Density with respect to a point mass at zero plus Lebesgue measure on positives.
pub fn hurdle_lognormal_logp(
    y: f64,
    probability: f64,
    log_level: f64,
    log_variance: f64,
) -> Result<f64, BayesianForecastError> {
    if !y.is_finite()
        || y < 0.0
        || !probability.is_finite()
        || !(0.0..=1.0).contains(&probability)
        || !log_level.is_finite()
        || !log_variance.is_finite()
        || log_variance <= 0.0
    {
        return Err(invalid("hurdle density needs nonnegative finite y, probability in [0,1], finite log level and positive log variance"));
    }
    if y == 0.0 {
        return Ok((-probability).ln_1p());
    }
    let log_y = y.ln();
    Ok(probability.ln()
        - log_y
        - 0.5
            * (std::f64::consts::TAU.ln()
                + log_variance.ln()
                + (log_y - log_level).powi(2) / log_variance))
}

pub fn fit_hurdle_lognormal(
    observations: &[f64],
    config: &HurdleLogNormalConfig,
) -> Result<HurdleLogNormalPosterior, BayesianForecastError> {
    config.validate()?;
    if observations.iter().any(|y| y.is_infinite() || *y < 0.0) {
        return Err(BayesianForecastError::InvalidObservations(
            "hurdle observations must be nonnegative finite values or NaN".into(),
        ));
    }
    let observed_count = observations.iter().filter(|y| y.is_finite()).count();
    let positive_count = observations.iter().filter(|y| **y > 0.0).count();
    if observed_count == 0 {
        return Err(BayesianForecastError::InvalidObservations(
            "at least one observed amount (including zero) is required".into(),
        ));
    }
    allocation(&[observations.len(), config.num_chains, 10])?;
    let occurrence = Beta::new(
        config.occurrence_alpha + positive_count as f64,
        config.occurrence_beta + (observed_count - positive_count) as f64,
    )
    .map_err(|e| numerical(e.to_string()))?;
    let log_observations: Vec<f64> = observations
        .iter()
        .map(|y| if *y > 0.0 { y.ln() } else { f64::NAN })
        .collect();
    let chains = (0..config.num_chains)
        .into_par_iter()
        .map(|chain| {
            let mut rng =
                ChaCha8Rng::seed_from_u64(chain_seed(config.seed, chain, 0x4855_5244_4649_5401));
            let mut draws = Vec::with_capacity(config.num_draws);
            // With no positive amounts the severity posterior equals its prior.
            // Direct independent draws avoid an uninformative augmented Gibbs chain.
            if positive_count == 0 {
                for _ in 0..config.num_draws {
                    let q = inverse_gamma(config.process_variance_prior, &mut rng)?;
                    let r = inverse_gamma(config.observation_variance_prior, &mut rng)?;
                    let variance = config.initial_variance + observations.len() as f64 * q;
                    let level = config.initial_log_level + normal(&mut rng) * variance.sqrt();
                    if !level.is_finite() {
                        return Err(numerical("prior terminal level overflowed"));
                    }
                    draws.push(HurdleLogNormalDraw {
                        payment_probability: occurrence.sample(&mut rng),
                        process_variance: q,
                        observation_variance: r,
                        terminal_log_level: level,
                    });
                }
                return Ok(draws);
            }
            let mut q =
                config.process_variance_prior.scale / (config.process_variance_prior.shape + 1.0);
            let mut r = config.observation_variance_prior.scale
                / (config.observation_variance_prior.shape + 1.0);
            let iterations = config.num_warmup + config.num_draws * config.thinning;
            for iteration in 0..iterations {
                let model = LinearGaussianStateSpace::local_level(
                    q,
                    r,
                    config.initial_log_level,
                    config.initial_variance,
                )
                .map_err(|e| numerical(e.to_string()))?;
                let states = model
                    .sample_states_ffbs(&log_observations, &mut rng)
                    .map_err(|e| numerical(e.to_string()))?;
                let process_ss: f64 = states
                    .windows(2)
                    .map(|pair| (pair[1][0] - pair[0][0]).powi(2))
                    .sum();
                q = inverse_gamma(
                    InverseGammaPrior {
                        shape: config.process_variance_prior.shape
                            + observations.len() as f64 / 2.0,
                        scale: config.process_variance_prior.scale + process_ss / 2.0,
                    },
                    &mut rng,
                )?;
                let observation_ss: f64 = log_observations
                    .iter()
                    .zip(&states[1..])
                    .filter(|(y, _)| y.is_finite())
                    .map(|(y, x)| (y - x[0]).powi(2))
                    .sum();
                r = inverse_gamma(
                    InverseGammaPrior {
                        shape: config.observation_variance_prior.shape
                            + positive_count as f64 / 2.0,
                        scale: config.observation_variance_prior.scale + observation_ss / 2.0,
                    },
                    &mut rng,
                )?;
                if iteration >= config.num_warmup
                    && (iteration + 1 - config.num_warmup).is_multiple_of(config.thinning)
                {
                    draws.push(HurdleLogNormalDraw {
                        payment_probability: occurrence.sample(&mut rng),
                        process_variance: q,
                        observation_variance: r,
                        terminal_log_level: states.last().expect("nonempty input")[0],
                    });
                }
            }
            Ok(draws)
        })
        .collect::<Result<Vec<_>, BayesianForecastError>>()?;
    Ok(HurdleLogNormalPosterior {
        chains,
        time_count: observations.len(),
        observed_count,
        positive_count,
    })
}

impl HurdleLogNormalPosterior {
    pub fn parameter_names() -> Vec<String> {
        [
            "payment_probability",
            "process_variance",
            "observation_variance",
            "terminal_log_level",
        ]
        .map(str::to_string)
        .to_vec()
    }

    pub fn parameter_samples(&self) -> Vec<Vec<Vec<f64>>> {
        self.chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        vec![
                            d.payment_probability,
                            d.process_variance,
                            d.observation_variance,
                            d.terminal_log_level,
                        ]
                    })
                    .collect()
            })
            .collect()
    }

    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<HurdleLogNormalForecast, BayesianForecastError> {
        if horizon == 0
            || self.chains.is_empty()
            || self.chains[0].is_empty()
            || self.chains.iter().any(|c| c.len() != self.chains[0].len())
        {
            return Err(invalid(
                "positive horizon and nonempty equal-length posterior chains required",
            ));
        }
        allocation(&[self.chains.len(), self.chains[0].len(), horizon, 3])?;
        type Paths = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);
        let chains: Vec<Paths> = self.chains.par_iter().enumerate().map(|(index, chain)| {
            let mut rng = ChaCha8Rng::seed_from_u64(chain_seed(seed, index, 0x4855_5244_5052_4501));
            let mut expected = Vec::with_capacity(chain.len());
            let mut positive = Vec::with_capacity(chain.len());
            let mut observed = Vec::with_capacity(chain.len());
            for draw in chain {
                if !draw.payment_probability.is_finite() || !(0.0..=1.0).contains(&draw.payment_probability)
                    || !draw.terminal_log_level.is_finite()
                    || !draw.process_variance.is_finite() || draw.process_variance <= 0.0
                    || !draw.observation_variance.is_finite() || draw.observation_variance <= 0.0 {
                    return Err(invalid("invalid hurdle posterior draw"));
                }
                let mut level = draw.terminal_log_level;
                let mut means = Vec::with_capacity(horizon);
                let mut positive_means = Vec::with_capacity(horizon);
                let mut observations = Vec::with_capacity(horizon);
                for _ in 0..horizon {
                    level += normal(&mut rng) * draw.process_variance.sqrt();
                    let positive_mean = (level + draw.observation_variance / 2.0).exp();
                    // Draw severity even when the indicator is zero so the two RNG
                    // streams' consumption does not depend on payment probability.
                    let amount = (level + normal(&mut rng) * draw.observation_variance.sqrt()).exp();
                    let paid = rng.gen::<f64>() < draw.payment_probability;
                    if !positive_mean.is_finite() || !amount.is_finite() || amount == 0.0 || positive_mean == 0.0 {
                        return Err(numerical("lognormal forecast overflowed or underflowed; inspect log-scale priors"));
                    }
                    means.push(draw.payment_probability * positive_mean);
                    positive_means.push(positive_mean);
                    observations.push(if paid { amount } else { 0.0 });
                }
                expected.push(means); positive.push(positive_means); observed.push(observations);
            }
            Ok((expected, positive, observed))
        }).collect::<Result<_, BayesianForecastError>>()?;
        let mut paths = PosteriorPredictiveForecast {
            state_paths: Vec::new(),
            observation_paths: Vec::new(),
        };
        let mut positive_mean_paths = Vec::new();
        for (means, positive, observed) in chains {
            paths.state_paths.push(means);
            paths.observation_paths.push(observed);
            positive_mean_paths.push(positive);
        }
        Ok(HurdleLogNormalForecast {
            paths,
            positive_mean_paths,
        })
    }
}

fn allocation(factors: &[usize]) -> Result<(), BayesianForecastError> {
    let count = factors
        .iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(*n));
    if count.is_none_or(|n| n > MAX_VALUES) {
        Err(invalid("requested hurdle allocation exceeds 25 million values; reduce chains, draws, history, or horizon"))
    } else {
        Ok(())
    }
}
fn inverse_gamma(
    prior: InverseGammaPrior,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    if !prior.shape.is_finite()
        || prior.shape <= 0.0
        || !prior.scale.is_finite()
        || prior.scale <= 0.0
    {
        return Err(numerical("invalid inverse-gamma conditional"));
    }
    let gamma = Gamma::new(prior.shape, 1.0 / prior.scale).map_err(|e| numerical(e.to_string()))?;
    let value = 1.0 / gamma.sample(rng);
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(numerical("inverse-gamma draw is nonfinite or nonpositive"))
    }
}
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}
fn chain_seed(seed: u64, chain: usize, domain: u64) -> u64 {
    let mut z = seed ^ domain ^ (chain as u64).wrapping_mul(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn invalid(message: impl Into<String>) -> BayesianForecastError {
    BayesianForecastError::InvalidConfiguration(message.into())
}
fn numerical(message: impl Into<String>) -> BayesianForecastError {
    BayesianForecastError::NumericalFailure(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> HurdleLogNormalConfig {
        HurdleLogNormalConfig {
            occurrence_alpha: 2.,
            occurrence_beta: 3.,
            initial_log_level: 1.,
            initial_variance: 0.2,
            process_variance_prior: InverseGammaPrior::new(4., 0.03).unwrap(),
            observation_variance_prior: InverseGammaPrior::new(4., 0.3).unwrap(),
            num_chains: 2,
            num_draws: 2000,
            num_warmup: 50,
            thinning: 1,
            seed: 92,
        }
    }
    #[test]
    fn occurrence_posterior_matches_beta_and_severity_prior_for_zero_history() {
        let fit = fit_hurdle_lognormal(&[0., 0., f64::NAN, 0.], &config()).unwrap();
        assert_eq!((fit.observed_count, fit.positive_count), (3, 0));
        let draws: Vec<_> = fit.chains.iter().flatten().collect();
        let mean = |f: fn(&HurdleLogNormalDraw) -> f64| {
            draws.iter().map(|d| f(d)).sum::<f64>() / draws.len() as f64
        };
        assert!((mean(|d| d.payment_probability) - 2. / 8.).abs() < 0.01);
        assert!((mean(|d| d.process_variance) - 0.01).abs() < 0.001);
        assert!((mean(|d| d.observation_variance) - 0.1).abs() < 0.008);
        assert!((mean(|d| d.terminal_log_level) - 1.).abs() < 0.025);
        let variance = draws
            .iter()
            .map(|d| (d.terminal_log_level - 1.).powi(2))
            .sum::<f64>()
            / draws.len() as f64;
        assert!((variance - 0.24).abs() < 0.02);
    }
    #[test]
    fn density_matches_point_mass_and_lognormal_jacobian() {
        assert!((hurdle_lognormal_logp(0., 0.3, 1., 0.5).unwrap() - 0.7_f64.ln()).abs() < 1e-14);
        let at_median = hurdle_lognormal_logp(1_f64.exp(), 0.3, 1., 0.5).unwrap();
        assert!((at_median - (0.3_f64.ln() - 1. - 0.5 * std::f64::consts::PI.ln())).abs() < 1e-14);
        assert_eq!(
            hurdle_lognormal_logp(0., 1., 0., 1.).unwrap(),
            f64::NEG_INFINITY
        );
        assert!(hurdle_lognormal_logp(-1., 0.5, 0., 1.).is_err());
    }
    #[test]
    fn one_positive_and_missing_steps_are_supported_and_seeded_across_threads() {
        let mut cfg = config();
        cfg.num_draws = 40;
        let y = [0., f64::NAN, 2., 0., 0.];
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let fit = fit_hurdle_lognormal(&y, &cfg).unwrap();
                    let f = fit.forecast(3, 99).unwrap();
                    (fit, f)
                })
        };
        assert_eq!(run(1), run(3));
    }
    #[test]
    fn predictive_zero_mass_and_amount_moments_match_known_parameters() {
        let draw = HurdleLogNormalDraw {
            payment_probability: 0.3,
            process_variance: 0.01,
            observation_variance: 0.2,
            terminal_log_level: 1.,
        };
        let fit = HurdleLogNormalPosterior {
            chains: vec![vec![draw; 30000]],
            time_count: 1,
            observed_count: 1,
            positive_count: 1,
        };
        let f = fit.forecast(1, 42).unwrap();
        let y: Vec<f64> = f.paths.observation_paths[0].iter().map(|p| p[0]).collect();
        let zero = y.iter().filter(|v| **v == 0.).count() as f64 / y.len() as f64;
        assert!((zero - 0.7).abs() < 0.01);
        let expected = 0.3 * (1.0_f64 + 0.21 / 2.).exp();
        assert!((y.iter().sum::<f64>() / y.len() as f64 - expected).abs() < 0.035);
        for (mean, pos) in f.paths.state_paths[0].iter().zip(&f.positive_mean_paths[0]) {
            assert_eq!(mean[0], 0.3 * pos[0]);
        }
    }
    #[test]
    fn rejects_invalid_observations_and_allocation_counts() {
        for y in [&[][..], &[f64::NAN], &[-1.], &[f64::INFINITY]] {
            assert!(fit_hurdle_lognormal(y, &config()).is_err());
        }
        let mut cfg = config();
        cfg.num_draws = usize::MAX;
        assert!(fit_hurdle_lognormal(&[0.], &cfg).is_err());
    }
}
