//! Conjugate Bayesian structural seasonal local-level inference.
//!
//! The latent state uses sum-to-zero dummy seasonality with one level and
//! `period - 1` seasonal states. Each time step has one level innovation and
//! one seasonal innovation; their variances and the observation variance have
//! inverse-gamma priors. Latent states are updated jointly with multivariate
//! FFBS, so missing observations retain their calendar positions.

use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::bayesian_forecast::{BayesianForecastError, ForecastQuantile, InverseGammaPrior};
use crate::forecast_common::{
    check_forecast_size, overdispersed_positive, path_means, path_quantiles,
    require_finite_observations, run_gibbs_chains, sample_inverse_gamma, simulate_draws,
    split_paths, GibbsSchedule,
};
use crate::state_space::LinearGaussianStateSpace;

#[derive(Debug, Clone)]
pub struct BayesianSeasonalLocalLevelConfig {
    pub period: usize,
    pub initial_level: f64,
    pub initial_seasonal_effects: Vec<f64>,
    pub initial_level_variance: f64,
    pub initial_seasonal_variance: f64,
    pub level_variance_prior: InverseGammaPrior,
    pub seasonal_variance_prior: InverseGammaPrior,
    pub observation_variance_prior: InverseGammaPrior,
    pub num_chains: usize,
    pub num_warmup: usize,
    pub num_draws: usize,
    pub thinning: usize,
    pub seed: u64,
}

impl BayesianSeasonalLocalLevelConfig {
    fn validate(&self) -> Result<GibbsSchedule, BayesianForecastError> {
        if self.period < 2 {
            return Err(invalid("seasonal period must be at least 2"));
        }
        if self.initial_seasonal_effects.len() != self.period {
            return Err(invalid(&format!(
                "initial seasonal effects have {} entries; expected {}",
                self.initial_seasonal_effects.len(),
                self.period
            )));
        }
        if !self.initial_level.is_finite()
            || self
                .initial_seasonal_effects
                .iter()
                .any(|effect| !effect.is_finite())
        {
            return Err(invalid("initial level and seasonal effects must be finite"));
        }
        let scale = self
            .initial_seasonal_effects
            .iter()
            .map(|effect| effect.abs())
            .sum::<f64>()
            .max(1.0);
        let sum = self.initial_seasonal_effects.iter().sum::<f64>();
        if sum.abs() > 1e-10 * scale {
            return Err(invalid("initial seasonal effects must sum to zero"));
        }
        if !self.initial_level_variance.is_finite() || self.initial_level_variance <= 0.0 {
            return Err(invalid(
                "initial level variance must be finite and strictly positive",
            ));
        }
        if !self.initial_seasonal_variance.is_finite() || self.initial_seasonal_variance <= 0.0 {
            return Err(invalid(
                "initial seasonal variance must be finite and strictly positive",
            ));
        }
        for (name, prior) in [
            ("level-variance prior", self.level_variance_prior),
            ("seasonal-variance prior", self.seasonal_variance_prior),
            (
                "observation-variance prior",
                self.observation_variance_prior,
            ),
        ] {
            prior.validate(name)?;
        }
        GibbsSchedule::new(
            self.num_chains,
            self.num_warmup,
            self.num_draws,
            self.thinning,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalLocalLevelPosteriorDraw {
    pub level_variance: f64,
    pub seasonal_variance: f64,
    pub observation_variance: f64,
    /// State at the final observed time: level followed by seasonal history.
    pub terminal_state: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalLocalLevelPosterior {
    pub period: usize,
    pub chains: Vec<Vec<SeasonalLocalLevelPosteriorDraw>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalPosteriorPredictiveForecast {
    pub level_paths: Vec<Vec<Vec<f64>>>,
    pub seasonal_paths: Vec<Vec<Vec<f64>>>,
    pub observation_paths: Vec<Vec<Vec<f64>>>,
    pub cumulative_observation_paths: Vec<Vec<Vec<f64>>>,
}

impl SeasonalLocalLevelPosterior {
    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<SeasonalPosteriorPredictiveForecast, BayesianForecastError> {
        if horizon == 0 {
            return Err(invalid("forecast horizon must be positive"));
        }
        if self.period < 2 || self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(invalid(
                "posterior must contain a valid period and non-empty chains",
            ));
        }
        check_forecast_size("seasonal forecast", &self.chains, horizon, 4)?;
        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &SeasonalLocalLevelPosteriorDraw, rng| {
                validate_variance("level", draw.level_variance)?;
                validate_variance("seasonal", draw.seasonal_variance)?;
                validate_variance("observation", draw.observation_variance)?;
                if draw.terminal_state.len() != self.period
                    || draw.terminal_state.iter().any(|value| !value.is_finite())
                {
                    return Err(numerical("posterior terminal seasonal state is invalid"));
                }
                let mut state = draw.terminal_state.clone();
                let mut levels = Vec::with_capacity(horizon);
                let mut seasonals = Vec::with_capacity(horizon);
                let mut observations = Vec::with_capacity(horizon);
                let mut cumulative = Vec::with_capacity(horizon);
                let mut running_sum = 0.0;
                for _ in 0..horizon {
                    let next_level = state[0] + standard_normal(rng) * draw.level_variance.sqrt();
                    let next_seasonal = -state[1..].iter().sum::<f64>()
                        + standard_normal(rng) * draw.seasonal_variance.sqrt();
                    for index in (2..self.period).rev() {
                        state[index] = state[index - 1];
                    }
                    state[0] = next_level;
                    state[1] = next_seasonal;
                    let observation = next_level
                        + next_seasonal
                        + standard_normal(rng) * draw.observation_variance.sqrt();
                    running_sum += observation;
                    if !observation.is_finite() || !running_sum.is_finite() {
                        return Err(numerical("seasonal forecast simulation overflowed"));
                    }
                    levels.push(next_level);
                    seasonals.push(next_seasonal);
                    observations.push(observation);
                    cumulative.push(running_sum);
                }
                Ok([levels, seasonals, observations, cumulative])
            },
        )?;
        let [level_paths, seasonal_paths, observation_paths, cumulative_observation_paths] =
            split_paths(per_draw);
        Ok(SeasonalPosteriorPredictiveForecast {
            level_paths,
            seasonal_paths,
            observation_paths,
            cumulative_observation_paths,
        })
    }
}

impl SeasonalPosteriorPredictiveForecast {
    pub fn horizon(&self) -> usize {
        self.observation_paths
            .first()
            .and_then(|chain| chain.first())
            .map_or(0, Vec::len)
    }

    pub fn level_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.level_paths)
    }
    pub fn seasonal_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.seasonal_paths)
    }
    pub fn observation_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.observation_paths)
    }
    pub fn cumulative_observation_means(&self) -> Result<Vec<f64>, BayesianForecastError> {
        path_means(&self.cumulative_observation_paths)
    }
    pub fn level_quantiles(
        &self,
        p: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.level_paths, p)
    }
    pub fn seasonal_quantiles(
        &self,
        p: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.seasonal_paths, p)
    }
    pub fn observation_quantiles(
        &self,
        p: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.observation_paths, p)
    }
    pub fn cumulative_observation_quantiles(
        &self,
        p: &[f64],
    ) -> Result<Vec<ForecastQuantile>, BayesianForecastError> {
        path_quantiles(&self.cumulative_observation_paths, p)
    }
}

pub fn fit_bayesian_seasonal_local_level(
    observations: &[f64],
    config: &BayesianSeasonalLocalLevelConfig,
) -> Result<SeasonalLocalLevelPosterior, BayesianForecastError> {
    let schedule = config.validate()?;
    let observed_count = validate_observations(observations)? as f64;
    let transitions = observations.len() as f64;
    // Built and validated once; each sweep overwrites only the level (0) and
    // seasonal (1) innovation variances and the observation variance.
    let template = LinearGaussianStateSpace::seasonal_local_level(
        config.period,
        config.level_variance_prior.mode(),
        config.seasonal_variance_prior.mode(),
        config.observation_variance_prior.mode(),
        config.initial_level,
        config.initial_seasonal_effects.clone(),
        config.initial_level_variance,
        config.initial_seasonal_variance,
    )
    .map_err(|error| numerical(&format!("could not build seasonal state model: {error}")))?;
    let chains = run_gibbs_chains(
        &schedule,
        config.seed,
        FIT_SEED_DOMAIN,
        |rng| {
            Ok::<_, BayesianForecastError>((
                template.clone(),
                [
                    overdispersed_positive(config.level_variance_prior.mode(), rng),
                    overdispersed_positive(config.seasonal_variance_prior.mode(), rng),
                    overdispersed_positive(config.observation_variance_prior.mode(), rng),
                ],
            ))
        },
        |(model, [level_variance, seasonal_variance, observation_variance]), rng, retain| {
            model.set_variances(
                &[0, 1],
                &[*level_variance, *seasonal_variance],
                *observation_variance,
            );
            // states[0] is x[-1], followed by x[0]..x[T-1].
            let states = model
                .sample_states_ffbs(observations, rng)
                .map_err(|error| numerical(&format!("seasonal FFBS failed: {error}")))?;

            let mut level_sum_sq = 0.0;
            let mut seasonal_sum_sq = 0.0;
            for pair in states.windows(2) {
                let level_residual = pair[1][0] - pair[0][0];
                let seasonal_residual = pair[1][1] + pair[0][1..].iter().sum::<f64>();
                level_sum_sq += level_residual * level_residual;
                seasonal_sum_sq += seasonal_residual * seasonal_residual;
            }
            *level_variance = sample_inverse_gamma(
                config.level_variance_prior.shape + transitions / 2.0,
                config.level_variance_prior.scale + level_sum_sq / 2.0,
                rng,
            )?;
            *seasonal_variance = sample_inverse_gamma(
                config.seasonal_variance_prior.shape + transitions / 2.0,
                config.seasonal_variance_prior.scale + seasonal_sum_sq / 2.0,
                rng,
            )?;
            let observation_sum_sq: f64 = observations
                .iter()
                .zip(&states[1..])
                .filter(|(observation, _)| !observation.is_nan())
                .map(|(observation, state)| {
                    let residual = observation - state[0] - state[1];
                    residual * residual
                })
                .sum();
            *observation_variance = sample_inverse_gamma(
                config.observation_variance_prior.shape + observed_count / 2.0,
                config.observation_variance_prior.scale + observation_sum_sq / 2.0,
                rng,
            )?;

            Ok(retain.then(|| SeasonalLocalLevelPosteriorDraw {
                level_variance: *level_variance,
                seasonal_variance: *seasonal_variance,
                observation_variance: *observation_variance,
                terminal_state: states.last().expect("observations are non-empty").clone(),
            }))
        },
    )?;
    Ok(SeasonalLocalLevelPosterior {
        period: config.period,
        chains,
    })
}

fn validate_observations(observations: &[f64]) -> Result<usize, BayesianForecastError> {
    if observations.iter().any(|value| value.is_infinite()) {
        return Err(BayesianForecastError::InvalidObservations(
            "observations may be finite or NaN, but not infinite".into(),
        ));
    }
    require_finite_observations(observations, 2, "seasonal local-level")
}

fn validate_variance(name: &str, variance: f64) -> Result<(), BayesianForecastError> {
    if !variance.is_finite() || variance <= 0.0 {
        return Err(numerical(&format!(
            "{name} variance must be finite and strictly positive"
        )));
    }
    Ok(())
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

fn invalid(message: &str) -> BayesianForecastError {
    BayesianForecastError::InvalidConfiguration(message.into())
}

fn numerical(message: &str) -> BayesianForecastError {
    BayesianForecastError::NumericalFailure(message.into())
}

const FIT_SEED_DOMAIN: u64 = 0x5345_4153_5F46_4954;
const FORECAST_SEED_DOMAIN: u64 = 0x5345_4153_5F46_4353;

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BayesianSeasonalLocalLevelConfig {
        BayesianSeasonalLocalLevelConfig {
            period: 4,
            initial_level: 5.0,
            initial_seasonal_effects: vec![1.0, -0.5, -0.25, -0.25],
            initial_level_variance: 4.0,
            initial_seasonal_variance: 2.0,
            level_variance_prior: InverseGammaPrior::new(3.0, 0.2).unwrap(),
            seasonal_variance_prior: InverseGammaPrior::new(3.0, 0.1).unwrap(),
            observation_variance_prior: InverseGammaPrior::new(3.0, 0.5).unwrap(),
            num_chains: 2,
            num_warmup: 30,
            num_draws: 40,
            thinning: 1,
            seed: 7,
        }
    }

    #[test]
    fn fit_and_forecast_are_seeded_coherent_and_support_missing_values() {
        let observations = [
            6.0,
            4.6,
            4.8,
            4.9,
            6.1,
            f64::NAN,
            4.7,
            5.0,
            6.2,
            4.7,
            4.9,
            5.1,
        ];
        let first = fit_bayesian_seasonal_local_level(&observations, &config()).unwrap();
        let second = fit_bayesian_seasonal_local_level(&observations, &config()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.chains.len(), 2);
        assert!(first.chains.iter().all(|chain| chain.len() == 40));
        for draw in first.chains.iter().flatten() {
            assert!(draw.level_variance > 0.0);
            assert!(draw.seasonal_variance > 0.0);
            assert!(draw.observation_variance > 0.0);
            assert_eq!(draw.terminal_state.len(), 4);
        }

        let forecast = first.forecast(8, 9).unwrap();
        assert_eq!(forecast, first.forecast(8, 9).unwrap());
        assert_eq!(forecast.horizon(), 8);
        for chain in &forecast.observation_paths {
            assert!(chain.iter().all(|path| path.len() == 8));
        }
        for (observations, cumulative) in forecast
            .observation_paths
            .iter()
            .flatten()
            .zip(forecast.cumulative_observation_paths.iter().flatten())
        {
            let mut running = 0.0;
            for (observation, cumulative) in observations.iter().zip(cumulative) {
                running += observation;
                assert_eq!(*cumulative, running);
            }
        }
    }

    #[test]
    fn validation_requires_finite_data_and_sum_to_zero_effects() {
        assert!(fit_bayesian_seasonal_local_level(&[0.0; 2], &config()).is_ok());
        let mut invalid_config = config();
        invalid_config.initial_seasonal_effects[0] += 1.0;
        assert!(matches!(
            fit_bayesian_seasonal_local_level(&[0.0; 12], &invalid_config),
            Err(BayesianForecastError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            fit_bayesian_seasonal_local_level(&[0.0, f64::NAN], &config()),
            Err(BayesianForecastError::InvalidObservations(_))
        ));
    }
}
