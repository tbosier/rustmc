//! A Bernoulli hurdle with dynamic lognormal positive amounts.
//!
//! `P(y[t] > 0) = p`, `p ~ Beta(a,b)`, and conditional on a positive amount,
//! `log(y[t]) = level[t] + Normal(0,r)`, with a Gaussian random-walk level
//! and explicitly upper-truncated inverse-gamma priors on q and r. The finite
//! variance caps give finite positive predictive amount moments at finite horizons.
//! Zeros inform occurrence only;
//! missing values inform neither component, but both retain their time position.
//! Occurrence is static and independent of severity a priori. This factorization
//! permits exact Beta occurrence draws and conjugate Gaussian severity FFBS/Gibbs.

use crate::bayesian_forecast::{
    sample_levels_ffbs, BayesianForecastError, ForecastQuantile, InverseGammaPrior,
    PosteriorPredictiveForecast,
};
use crate::diagnostics::DiagnosticsReport;
use crate::forecast_common::{
    check_forecast_size, checked_value_count, inverse_gamma_conditional, overdispersed_positive,
    path_means, path_quantiles, run_chains, run_gibbs_chains, simulate_draws, split_paths,
    GibbsSchedule, MAX_MATERIALIZED_VALUES,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Distribution, StandardNormal};

const MAX_TRUNCATION_ATTEMPTS: usize = 100_000;
/// `"HURDFIT"` followed by a version byte.
const FIT_SEED_DOMAIN: u64 = 0x4855_5244_4649_5401;
/// `"HURDPRE"` followed by a version byte.
const FORECAST_SEED_DOMAIN: u64 = 0x4855_5244_5052_4501;

#[derive(Debug, Clone)]
pub struct HurdleLogNormalConfig {
    pub occurrence_alpha: f64,
    pub occurrence_beta: f64,
    pub initial_log_level: f64,
    pub initial_variance: f64,
    pub process_variance_prior: InverseGammaPrior,
    pub observation_variance_prior: InverseGammaPrior,
    /// Fixed upper support bound on process log variance, part of the prior.
    pub process_variance_upper: f64,
    /// Fixed upper support bound on observation log variance, part of the prior.
    pub observation_variance_upper: f64,
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub thinning: usize,
    pub seed: u64,
}

impl HurdleLogNormalConfig {
    pub fn validate(&self) -> Result<(), BayesianForecastError> {
        self.schedule().map(|_| ())
    }

    fn schedule(&self) -> Result<GibbsSchedule, BayesianForecastError> {
        for (name, value) in [
            ("occurrence alpha", self.occurrence_alpha),
            ("occurrence beta", self.occurrence_beta),
            ("initial variance", self.initial_variance),
            ("process variance upper bound", self.process_variance_upper),
            (
                "observation variance upper bound",
                self.observation_variance_upper,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!("{name} must be finite and positive")));
            }
        }
        if !self.initial_log_level.is_finite() {
            return Err(invalid("initial log level must be finite"));
        }
        if !(self.occurrence_alpha + self.occurrence_beta).is_finite() {
            return Err(invalid("occurrence prior shape sum must be finite"));
        }
        InverseGammaPrior::new(
            self.process_variance_prior.shape,
            self.process_variance_prior.scale,
        )?;
        InverseGammaPrior::new(
            self.observation_variance_prior.shape,
            self.observation_variance_prior.scale,
        )?;
        let schedule = GibbsSchedule::new(
            self.num_chains,
            self.num_warmup,
            self.num_draws,
            self.thinning,
        )?;
        checked_value_count(
            "hurdle posterior",
            &[self.num_chains, self.num_draws, 4],
            MAX_MATERIALIZED_VALUES,
        )?;
        Ok(schedule)
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
    /// `observation_paths` hold exact zeros and positive payment realizations.
    ///
    /// For this model `state_paths` are **not** latent states: they hold the
    /// conditional expected amount `E[y | p, future log level, observation
    /// variance]` along each path, so `paths.state_means()` and
    /// `paths.state_quantiles()` summarise that expectation. Prefer the
    /// [`expected_value_paths`](Self::expected_value_paths) family, which says
    /// what it returns; the field keeps its shared type so batch and Python
    /// code that reads it is unaffected.
    pub paths: PosteriorPredictiveForecast,
    /// E[y | y > 0, future log level, observation variance].
    pub positive_mean_paths: Vec<Vec<Vec<f64>>>,
}

impl HurdleLogNormalForecast {
    /// Conditional expected amounts `E[y | p, log level, variance]`, indexed
    /// `[chain][draw][step]`, including the probability of no payment.
    pub fn expected_value_paths(&self) -> &[Vec<Vec<f64>>] {
        &self.paths.state_paths
    }

    /// Posterior mean of the conditional expected amount at each step.
    pub fn expected_value_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.paths.state_paths)
    }

    /// Empirical quantiles of the conditional expected amount at each step.
    pub fn expected_value_quantiles(
        &self,
        probabilities: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.paths.state_paths, probabilities)
    }
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
    let schedule = config.schedule()?;
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
    checked_value_count(
        "hurdle working state",
        &[observations.len(), config.num_chains, 10],
        MAX_MATERIALIZED_VALUES,
    )?;
    let occurrence_alpha = config.occurrence_alpha + positive_count as f64;
    let occurrence_beta = config.occurrence_beta + (observed_count - positive_count) as f64;
    if !occurrence_alpha.is_finite()
        || !occurrence_beta.is_finite()
        || !(occurrence_alpha + occurrence_beta).is_finite()
    {
        return Err(numerical(
            "occurrence posterior shape arithmetic overflowed",
        ));
    }
    let occurrence =
        Beta::new(occurrence_alpha, occurrence_beta).map_err(|e| numerical(e.to_string()))?;
    let log_observations: Vec<f64> = observations
        .iter()
        .map(|y| if *y > 0.0 { y.ln() } else { f64::NAN })
        .collect();

    // With no positive amounts the severity posterior equals its prior.
    // Direct independent draws avoid an uninformative augmented Gibbs chain.
    let chains = if positive_count == 0 {
        run_chains(config.num_chains, config.seed, FIT_SEED_DOMAIN, |_, rng| {
            let mut draws = Vec::with_capacity(config.num_draws);
            for _ in 0..config.num_draws {
                let q = inverse_gamma(
                    config.process_variance_prior,
                    config.process_variance_upper,
                    rng,
                )?;
                let r = inverse_gamma(
                    config.observation_variance_prior,
                    config.observation_variance_upper,
                    rng,
                )?;
                let variance = config.initial_variance + observations.len() as f64 * q;
                let level = config.initial_log_level + normal(rng) * variance.sqrt();
                if !level.is_finite() {
                    return Err(numerical("prior terminal level overflowed"));
                }
                draws.push(HurdleLogNormalDraw {
                    payment_probability: probability_draw(&occurrence, rng)?,
                    process_variance: q,
                    observation_variance: r,
                    terminal_log_level: level,
                });
            }
            Ok(draws)
        })?
    } else {
        run_gibbs_chains(
            &schedule,
            config.seed,
            FIT_SEED_DOMAIN,
            // These are starting values, not clipped draws from either prior.
            |rng| {
                Ok::<_, BayesianForecastError>((
                    initial_variance(
                        config.process_variance_prior,
                        config.process_variance_upper,
                        rng,
                    )?,
                    initial_variance(
                        config.observation_variance_prior,
                        config.observation_variance_upper,
                        rng,
                    )?,
                ))
            },
            |(q, r), rng, retain| {
                // The severity is a scalar local level on log amounts, so it
                // uses the scalar FFBS; the general d-dimensional one gives the
                // same draws in distribution at about twenty times the cost.
                let levels = sample_levels_ffbs(
                    &log_observations,
                    config.initial_log_level,
                    config.initial_variance,
                    *q,
                    *r,
                    rng,
                )?;
                let process_ss: f64 = levels
                    .windows(2)
                    .map(|pair| (pair[1] - pair[0]).powi(2))
                    .sum();
                *q = inverse_gamma(
                    InverseGammaPrior {
                        shape: config.process_variance_prior.shape
                            + observations.len() as f64 / 2.0,
                        scale: config.process_variance_prior.scale + process_ss / 2.0,
                    },
                    config.process_variance_upper,
                    rng,
                )?;
                let observation_ss: f64 = log_observations
                    .iter()
                    .zip(&levels[1..])
                    .filter(|(y, _)| y.is_finite())
                    .map(|(y, x)| (y - x).powi(2))
                    .sum();
                *r = inverse_gamma(
                    InverseGammaPrior {
                        shape: config.observation_variance_prior.shape
                            + positive_count as f64 / 2.0,
                        scale: config.observation_variance_prior.scale + observation_ss / 2.0,
                    },
                    config.observation_variance_upper,
                    rng,
                )?;
                if !retain {
                    return Ok(None);
                }
                Ok(Some(HurdleLogNormalDraw {
                    payment_probability: probability_draw(&occurrence, rng)?,
                    process_variance: *q,
                    observation_variance: *r,
                    terminal_log_level: *levels.last().expect("nonempty input"),
                }))
            },
        )?
    };
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

    /// Rank-normalized R-hat, bulk/tail ESS, MCSE and HDIs for the occurrence
    /// probability, both variances and the terminal log level.
    pub fn diagnostics(&self) -> DiagnosticsReport {
        crate::forecast_diagnostics::parameter_diagnostics(
            &self.parameter_samples(),
            &Self::parameter_names(),
        )
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
        check_forecast_size("hurdle forecast", &self.chains, horizon, 3)?;
        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &HurdleLogNormalDraw, rng| {
                if !draw.payment_probability.is_finite()
                    || !(0.0..=1.0).contains(&draw.payment_probability)
                    || !draw.terminal_log_level.is_finite()
                    || !draw.process_variance.is_finite()
                    || draw.process_variance <= 0.0
                    || !draw.observation_variance.is_finite()
                    || draw.observation_variance <= 0.0
                {
                    return Err(invalid("invalid hurdle posterior draw"));
                }
                let mut level = draw.terminal_log_level;
                let mut means = Vec::with_capacity(horizon);
                let mut positive_means = Vec::with_capacity(horizon);
                let mut observations = Vec::with_capacity(horizon);
                for _ in 0..horizon {
                    level += normal(rng) * draw.process_variance.sqrt();
                    let positive_mean = (level + draw.observation_variance / 2.0).exp();
                    // Draw severity even when the indicator is zero so the two RNG
                    // streams' consumption does not depend on payment probability.
                    let amount = (level + normal(rng) * draw.observation_variance.sqrt()).exp();
                    let paid = rng.gen::<f64>() < draw.payment_probability;
                    if !positive_mean.is_finite()
                        || !amount.is_finite()
                        || amount == 0.0
                        || positive_mean == 0.0
                    {
                        return Err(numerical(
                            "lognormal forecast overflowed or underflowed; inspect log-scale priors",
                        ));
                    }
                    means.push(draw.payment_probability * positive_mean);
                    positive_means.push(positive_mean);
                    observations.push(if paid { amount } else { 0.0 });
                }
                Ok([means, positive_means, observations])
            },
        )?;
        let [state_paths, positive_mean_paths, observation_paths] = split_paths(per_draw);
        let paths = PosteriorPredictiveForecast {
            state_paths,
            observation_paths,
        };
        Ok(HurdleLogNormalForecast {
            paths,
            positive_mean_paths,
        })
    }
}

fn inverse_gamma(
    prior: InverseGammaPrior,
    upper: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    if !upper.is_finite() || upper <= 0.0 {
        return Err(numerical(
            "invalid upper-truncated inverse-gamma conditional",
        ));
    }
    let gamma = inverse_gamma_conditional(prior.shape, prior.scale)?;
    for _ in 0..MAX_TRUNCATION_ATTEMPTS {
        let precision = gamma.sample(rng);
        if !precision.is_finite() || precision < 0.0 {
            return Err(numerical(
                "inverse-gamma precision draw is nonfinite or negative",
            ));
        }
        let value = 1.0 / precision;
        // A zero precision represents a proposal beyond the upper support bound;
        // rejecting it also avoids turning a heavy-tail underflow into a sample.
        if value > upper {
            continue;
        }
        if value.is_finite() && value > 0.0 {
            return Ok(value);
        }
        return Err(numerical("inverse-gamma draw is nonfinite or nonpositive"));
    }
    Err(numerical(format!("upper-truncated inverse-gamma rejection exhausted {MAX_TRUNCATION_ATTEMPTS} attempts (shape={}, scale={}, upper={upper}); the fixed cap retains too little conditional mass; review the prior cap and data scale", prior.shape, prior.scale)))
}
/// The prior mode, kept inside the cap, spread over the shared log-uniform
/// start window and clipped back to the cap so the start is in support.
fn initial_variance(
    prior: InverseGammaPrior,
    upper: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    let value = overdispersed_positive(prior.mode().min(upper / 2.0), rng).min(upper);
    if !value.is_finite() || value <= 0.0 {
        return Err(numerical(
            "initial log variance inside its cap is unrepresentable",
        ));
    }
    Ok(value)
}
fn probability_draw(beta: &Beta<f64>, rng: &mut ChaCha8Rng) -> Result<f64, BayesianForecastError> {
    let value = beta.sample(rng);
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(numerical("occurrence probability draw is nonfinite or outside [0,1]; rescale extreme Beta concentrations"));
    }
    Ok(value)
}
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
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
    use crate::seeding::chain_seed;

    #[test]
    fn fit_and_forecast_seed_domains_are_distinct() {
        for chain in 0..4 {
            assert_ne!(
                chain_seed(42, chain, FIT_SEED_DOMAIN),
                chain_seed(42, chain, FORECAST_SEED_DOMAIN)
            );
        }
    }
    use rand::SeedableRng;
    use rand_distr::Gamma;
    /// `P(G >= t)` for `G ~ Gamma(k, 1)` at integer `k`, where the upper
    /// incomplete gamma closes in elementary terms. It gives `P(V <= v)` for
    /// `V ~ InverseGamma(k, scale)` exactly, at `t = scale / v`, because
    /// `V <= v` is `scale / V >= scale / v`.
    fn gamma_upper(k: u32, t: f64) -> f64 {
        let mut term = 1.0;
        let mut sum = 1.0;
        for i in 1..k {
            term *= t / i as f64;
            sum += term;
        }
        (-t).exp() * sum
    }

    /// `P(V <= v)` for `V ~ InverseGamma(shape, scale)` truncated to `(0, cap]`,
    /// which is the prior this module actually uses.
    fn truncated_inverse_gamma_cdf(prior: InverseGammaPrior, cap: f64, v: f64) -> f64 {
        let k = prior.shape as u32;
        gamma_upper(k, prior.scale / v.min(cap)) / gamma_upper(k, prior.scale / cap)
    }

    /// Mean of that truncated prior - the value a sampler that read no data
    /// would report. `E[V | V <= cap] = scale / (shape - 1) * Q(shape - 1, t) /
    /// Q(shape, t)` with `t = scale / cap`, from `E[V 1{V <= cap}] = scale /
    /// (shape - 1) * Q(shape - 1, t)`.
    fn truncated_inverse_gamma_mean(prior: InverseGammaPrior, cap: f64) -> f64 {
        let k = prior.shape as u32;
        let t = prior.scale / cap;
        prior.scale / (prior.shape - 1.0) * gamma_upper(k - 1, t) / gamma_upper(k, t)
    }

    /// `P(P <= p)` for `P ~ Beta(2, 3)`, whose density is `12 p (1 - p)^2`.
    fn beta23_cdf(p: f64) -> f64 {
        6.0 * p * p - 8.0 * p.powi(3) + 3.0 * p.powi(4)
    }

    /// A recovery window has to be one the prior cannot satisfy on its own: the
    /// prior mean has to sit at least a full tolerance-width outside it, and the
    /// prior has to place little mass inside it. Both bounds are permanent, so
    /// widening a window back onto the prior turns the test red.
    fn assert_window_beats_prior(
        name: &str,
        prior_mean: f64,
        prior_mass: f64,
        truth: f64,
        tolerance: f64,
        max_prior_mass: f64,
    ) {
        let widths = ((prior_mean - truth).abs() - tolerance) / tolerance;
        assert!(
            widths >= 1.0,
            "vacuous window for {name}: [{}, {}] lies only {widths:.2} tolerance-widths \
             from the prior mean {prior_mean}",
            truth - tolerance,
            truth + tolerance
        );
        assert!(
            prior_mass < max_prior_mass,
            "vacuous window for {name}: the prior places {prior_mass:.4} of its mass inside \
             [{}, {}], above the {max_prior_mass} this test is allowed",
            truth - tolerance,
            truth + tolerance
        );
    }

    fn config() -> HurdleLogNormalConfig {
        HurdleLogNormalConfig {
            occurrence_alpha: 2.,
            occurrence_beta: 3.,
            initial_log_level: 1.,
            initial_variance: 0.2,
            process_variance_prior: InverseGammaPrior::new(4., 0.03).unwrap(),
            observation_variance_prior: InverseGammaPrior::new(4., 0.3).unwrap(),
            process_variance_upper: 1.0,
            observation_variance_upper: 4.0,
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
    fn diagnostics_cover_every_parameter_and_report_mixing() {
        let mut cfg = config();
        cfg.num_draws = 300;
        cfg.num_warmup = 100;
        let fit = fit_hurdle_lognormal(&[0., 2., 3., f64::NAN, 0., 2.5, 1.5, 0.], &cfg).unwrap();
        let report = fit.diagnostics();
        let names: Vec<_> = report.params.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names, HurdleLogNormalPosterior::parameter_names());
        for parameter in &report.params {
            assert!(
                parameter.r_hat.is_finite() && parameter.r_hat < 1.1,
                "{parameter:?}"
            );
            assert!(parameter.ess_bulk > 50.0, "{parameter:?}");
        }
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
        assert_eq!(f.expected_value_paths(), &f.paths.state_paths[..]);
        assert_eq!(
            f.expected_value_means().unwrap(),
            f.paths.state_means().unwrap()
        );
        assert_eq!(
            f.expected_value_quantiles(&[0.1, 0.9]).unwrap(),
            f.paths.state_quantiles(&[0.1, 0.9]).unwrap()
        );
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

    #[test]
    fn truncated_prior_matches_analytic_mean_cdf_and_terminal_variance() {
        let mut cfg = config();
        cfg.process_variance_prior = InverseGammaPrior::new(2., 0.2).unwrap();
        cfg.observation_variance_prior = InverseGammaPrior::new(2., 0.8).unwrap();
        cfg.process_variance_upper = 0.1;
        cfg.observation_variance_upper = 0.4;
        cfg.num_draws = 10000;
        let fit = fit_hurdle_lognormal(&[0., f64::NAN, 0.], &cfg).unwrap();
        let samples: Vec<_> = fit.chains.iter().flatten().collect();
        let n = samples.len() as f64;
        // X~IG(2,b), X<=c: P(X<=c)=exp(-b/c)(1+b/c),
        // and E[X | X<=c] = b/(1+b/c).
        let q_mean = samples.iter().map(|d| d.process_variance).sum::<f64>() / n;
        let r_mean = samples.iter().map(|d| d.observation_variance).sum::<f64>() / n;
        assert!((q_mean - 0.2 / 3.).abs() < 0.001);
        assert!((r_mean - 0.8 / 3.).abs() < 0.003);
        let q_cdf = samples
            .iter()
            .filter(|d| d.process_variance <= 0.05)
            .count() as f64
            / n;
        assert!((q_cdf - (-2_f64).exp() * 5. / 3.).abs() < 0.012);
        assert!(samples
            .iter()
            .all(|d| d.process_variance < 0.1 && d.observation_variance < 0.4));
        let variance = samples
            .iter()
            .map(|d| (d.terminal_log_level - 1.).powi(2))
            .sum::<f64>()
            / n;
        assert!((variance - (0.2 + 3. * 0.2 / 3.)).abs() < 0.015);
        // The integrated arithmetic mean is finite under the capped priors.
        // Independently integrate their bounded exponential moments by quadrature.
        let exponential_moment = |scale: f64, cap: f64, multiplier: f64| {
            let mut weight_sum = 0.;
            let mut moment_sum = 0.;
            for i in 0..20000 {
                let x = (i as f64 + 0.5) * cap / 20000.;
                let weight = (-3. * x.ln() - scale / x).exp();
                weight_sum += weight;
                moment_sum += weight * (multiplier * x).exp();
            }
            moment_sum / weight_sum
        };
        let expected = (2. / 7.)
            * 1.1_f64.exp()
            * exponential_moment(0.2, 0.1, 2.)
            * exponential_moment(0.8, 0.4, 0.5);
        let forecast = fit.forecast(1, 561).unwrap();
        let actual = forecast
            .paths
            .observation_paths
            .iter()
            .flatten()
            .map(|d| d[0])
            .sum::<f64>()
            / n;
        assert!(
            (actual - expected).abs() < 0.055,
            "integrated mean {actual} vs {expected}"
        );
    }

    #[test]
    fn one_positive_matches_independent_truncated_prior_importance_integral() {
        let mut cfg = config();
        cfg.process_variance_prior = InverseGammaPrior::new(4., 0.3).unwrap();
        cfg.observation_variance_prior = InverseGammaPrior::new(4., 0.6).unwrap();
        cfg.process_variance_upper = 0.15;
        cfg.observation_variance_upper = 0.3;
        cfg.num_draws = 20000;
        cfg.num_warmup = 1000;
        // The single observation is exp(3), not exp(1.7). At exp(1.7) the log
        // residual against the initial level is 0.7 and the importance integral
        // returns 0.079097 and 0.158390 - within a thousandth of the truncated
        // prior means 0.078947 and 0.157895, and so inside the tolerances below.
        // The reference and the fit agreed because neither had moved. A residual
        // of 2.0 pulls them to 0.086437 and 0.188862; the negative control after
        // the loop keeps any future weakening honest.
        let fit = fit_hurdle_lognormal(&[3.0_f64.exp()], &cfg).unwrap();
        let mut actual = [0.; 3];
        for d in fit.chains.iter().flatten() {
            actual[0] += d.process_variance;
            actual[1] += d.observation_variance;
            actual[2] += d.terminal_log_level;
            assert!(d.process_variance <= cfg.process_variance_upper);
            assert!(d.observation_variance <= cfg.observation_variance_upper);
        }
        for mean in &mut actual {
            *mean /= (cfg.num_chains * cfg.num_draws) as f64;
        }
        // Integrate out both initial and observed states independently of FFBS:
        // log(y) | q,r ~ N(initial_mean, initial_variance + q + r).
        let mut rng = ChaCha8Rng::seed_from_u64(814);
        let gamma = Gamma::new(4., 1.).unwrap();
        let mut weighted = [0.; 4];
        for _ in 0..500000 {
            let q = 0.3 / gamma.sample(&mut rng);
            let r = 0.6 / gamma.sample(&mut rng);
            if q > cfg.process_variance_upper || r > cfg.observation_variance_upper {
                continue;
            }
            let variance: f64 = 0.2 + q + r;
            let weight = (-2.0_f64.powi(2) / (2. * variance)).exp() / variance.sqrt();
            weighted[0] += weight;
            weighted[1] += weight * q;
            weighted[2] += weight * r;
            weighted[3] += weight * (1. + (0.2 + q) / variance * 2.0);
        }
        // What a fit that never updated each parameter would report, which the
        // tolerances have to exclude: the truncated prior mean for the two
        // variances, and the starting value for the level.
        //
        // This is the weakest of the data-blind counterfactuals, not the
        // strongest. A fit that drew q and r from their priors and then did
        // update the level would report about 2.2969 against the reference
        // 2.2218 below, so the level's 0.015 tolerance separates them but a
        // tolerance above roughly 0.075 would not, while the guard here would
        // still pass. Ruling that one out needs the level integrated over the
        // variance priors by quadrature, which is more machinery than this
        // assertion earns; the bound it does enforce is stated rather than
        // implied.
        let uninformed = [
            truncated_inverse_gamma_mean(cfg.process_variance_prior, cfg.process_variance_upper),
            truncated_inverse_gamma_mean(
                cfg.observation_variance_prior,
                cfg.observation_variance_upper,
            ),
            cfg.initial_log_level,
        ];
        for (i, tolerance) in [0.002, 0.003, 0.015].iter().enumerate() {
            let expected = weighted[i + 1] / weighted[0];
            assert!(
                (expected - uninformed[i]).abs() > 2.0 * *tolerance,
                "parameter {i}: the reference {expected} is within two tolerances of \
                 {}, what a fit that never updated it would report",
                uninformed[i]
            );
            assert!(
                (actual[i] - expected).abs() < *tolerance,
                "parameter {i}: {} vs {expected}",
                actual[i]
            );
        }
    }

    #[test]
    fn invalid_caps_extreme_beta_and_unattainable_truncation_fail_explicitly() {
        for upper in [0., -1., f64::NAN, f64::INFINITY] {
            let mut cfg = config();
            cfg.process_variance_upper = upper;
            assert!(fit_hurdle_lognormal(&[0.], &cfg).is_err());
            cfg = config();
            cfg.observation_variance_upper = upper;
            assert!(fit_hurdle_lognormal(&[1.], &cfg).is_err());
        }
        let mut cfg = config();
        cfg.occurrence_alpha = 1e308;
        cfg.occurrence_beta = 1e308;
        assert!(fit_hurdle_lognormal(&[0.], &cfg).is_err());
        let beta = Beta::new(1e308, 1e308).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        assert!(probability_draw(&beta, &mut rng).is_err());
        let error =
            inverse_gamma(InverseGammaPrior::new(2., 100.).unwrap(), 0.0001, &mut rng).unwrap_err();
        assert!(error.to_string().contains("rejection exhausted"));
        assert!(inverse_gamma(
            InverseGammaPrior::new(2., f64::from_bits(1)).unwrap(),
            1.,
            &mut rng
        )
        .is_err());
    }

    #[test]
    fn dynamic_positive_severity_and_occurrence_recover_simulated_parameters() {
        let mut rng = ChaCha8Rng::seed_from_u64(604);
        let mut level = 1.;
        let mut observations = Vec::new();
        for _ in 0..1200 {
            level += normal(&mut rng) * 0.01_f64.sqrt();
            let amount = (level + normal(&mut rng) * 0.1_f64.sqrt()).exp();
            observations.push(if rng.gen::<f64>() < 0.6 { amount } else { 0. });
        }
        let mut cfg = config();
        cfg.num_draws = 1500;
        cfg.num_warmup = 750;
        cfg.process_variance_upper = 0.1;
        cfg.observation_variance_upper = 0.5;
        // The shared `config()` priors have means of exactly 0.01 and 0.1, the
        // two variances simulated above, so two of this test's three assertions
        // were satisfied by the prior alone whatever the sampler did with the
        // data. `occurrence_posterior_matches_beta_and_severity_prior_for_zero_history`
        // needs those priors and keeps them; here they are replaced by priors
        // whose truncated means are 0.05 and 0.333, five and three times the
        // truth, and which are flatter than the ones they replace (shape 2
        // rather than 4) so that the offset costs accuracy rather than buying
        // it. The series is also longer - 1200 steps rather than 320 - because a
        // process variance a tenth of the observation variance needs the length
        // to separate from it.
        cfg.process_variance_prior = InverseGammaPrior::new(2., 0.1).unwrap();
        cfg.observation_variance_prior = InverseGammaPrior::new(2., 1.0).unwrap();
        let fit = fit_hurdle_lognormal(&observations, &cfg).unwrap();
        let n = (cfg.num_chains * cfg.num_draws) as f64;
        let p = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.payment_probability)
            .sum::<f64>()
            / n;
        let q = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.process_variance)
            .sum::<f64>()
            / n;
        let r = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.observation_variance)
            .sum::<f64>()
            / n;
        const OCCURRENCE_TOLERANCE: f64 = 0.05;
        const PROCESS_TOLERANCE: f64 = 0.006;
        const OBSERVATION_TOLERANCE: f64 = 0.03;
        assert_window_beats_prior(
            "occurrence probability",
            cfg.occurrence_alpha / (cfg.occurrence_alpha + cfg.occurrence_beta),
            beta23_cdf(0.6 + OCCURRENCE_TOLERANCE) - beta23_cdf(0.6 - OCCURRENCE_TOLERANCE),
            0.6,
            OCCURRENCE_TOLERANCE,
            0.2,
        );
        assert_window_beats_prior(
            "process variance",
            truncated_inverse_gamma_mean(cfg.process_variance_prior, cfg.process_variance_upper),
            truncated_inverse_gamma_cdf(
                cfg.process_variance_prior,
                cfg.process_variance_upper,
                0.01 + PROCESS_TOLERANCE,
            ) - truncated_inverse_gamma_cdf(
                cfg.process_variance_prior,
                cfg.process_variance_upper,
                0.01 - PROCESS_TOLERANCE,
            ),
            0.01,
            PROCESS_TOLERANCE,
            0.02,
        );
        assert_window_beats_prior(
            "observation variance",
            truncated_inverse_gamma_mean(
                cfg.observation_variance_prior,
                cfg.observation_variance_upper,
            ),
            truncated_inverse_gamma_cdf(
                cfg.observation_variance_prior,
                cfg.observation_variance_upper,
                0.1 + OBSERVATION_TOLERANCE,
            ) - truncated_inverse_gamma_cdf(
                cfg.observation_variance_prior,
                cfg.observation_variance_upper,
                0.1 - OBSERVATION_TOLERANCE,
            ),
            0.1,
            OBSERVATION_TOLERANCE,
            0.02,
        );
        assert!((p - 0.6).abs() < OCCURRENCE_TOLERANCE, "occurrence {p}");
        assert!((q - 0.01).abs() < PROCESS_TOLERANCE, "process variance {q}");
        assert!(
            (r - 0.1).abs() < OBSERVATION_TOLERANCE,
            "observation variance {r}"
        );
        let forecast = fit.forecast(3, 903).unwrap();
        assert!(forecast
            .paths
            .observation_paths
            .iter()
            .flatten()
            .flatten()
            .all(|x| x.is_finite() && *x >= 0.));
    }
}
