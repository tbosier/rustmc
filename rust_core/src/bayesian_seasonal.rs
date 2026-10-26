//! Conjugate Bayesian structural seasonal local-level inference.
//!
//! The latent state uses sum-to-zero dummy seasonality with one level and
//! `period - 1` seasonal states. Each time step has one level innovation and
//! one seasonal innovation; their variances and the observation variance have
//! inverse-gamma priors. Latent states are updated jointly with multivariate
//! FFBS, so missing observations retain their calendar positions.

use crate::diagnostics::warnings::slow_state_variance_mixing;
use crate::diagnostics::{fit_warnings, DiagnosticsReport, FitWarning, SamplerTelemetry};
use crate::forecast_common::{
    check_forecast_size, invalid_config, numerical, overdispersed_positive, path_means,
    path_quantiles, require_finite_observations, run_gibbs_chains, simulate_draws, split_paths,
    standard_normal, validate_positive_variance, BayesianForecastError, ForecastQuantile,
    GibbsSchedule, InverseGammaPrior, SEASONAL_FIT_SEED_DOMAIN as FIT_SEED_DOMAIN,
    SEASONAL_FORECAST_SEED_DOMAIN as FORECAST_SEED_DOMAIN,
};
use crate::forecast_diagnostics::parameter_diagnostics;
use crate::state_space::LinearGaussianStateSpace;

/// How fit warnings name this model's sampler.
const SAMPLER_NAME: &str = "conjugate Gibbs/FFBS";

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
            return Err(invalid_config("seasonal period must be at least 2"));
        }
        if self.initial_seasonal_effects.len() != self.period {
            return Err(invalid_config(format!(
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
            return Err(invalid_config(
                "initial level and seasonal effects must be finite",
            ));
        }
        let scale = self
            .initial_seasonal_effects
            .iter()
            .map(|effect| effect.abs())
            .sum::<f64>()
            .max(1.0);
        let sum = self.initial_seasonal_effects.iter().sum::<f64>();
        if sum.abs() > 1e-10 * scale {
            return Err(invalid_config("initial seasonal effects must sum to zero"));
        }
        if !self.initial_level_variance.is_finite() || self.initial_level_variance <= 0.0 {
            return Err(invalid_config(
                "initial level variance must be finite and strictly positive",
            ));
        }
        if !self.initial_seasonal_variance.is_finite() || self.initial_seasonal_variance <= 0.0 {
            return Err(invalid_config(
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
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let mut names = vec![
            "level_variance".into(),
            "seasonal_variance".into(),
            "observation_variance".into(),
            "terminal_level".into(),
        ];
        names.extend((1..self.period).map(|i| format!("terminal_seasonal[{}]", i - 1)));
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        let mut values = vec![
                            d.level_variance,
                            d.seasonal_variance,
                            d.observation_variance,
                        ];
                        values.extend_from_slice(&d.terminal_state);
                        values
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }

    /// Every failed or unavailable check of this fit (see
    /// [`fit_warnings`]), followed by a `known_limitation` record for the
    /// slow mixing of the level and seasonal variances, which every seasonal
    /// local-level fit carries whatever its diagnostics show.
    pub fn warnings(&self) -> Vec<FitWarning> {
        let mut warnings = fit_warnings(
            &self.diagnostics(),
            SamplerTelemetry::NonHamiltonian {
                sampler: SAMPLER_NAME,
                constant: &[],
            },
        );
        // Range measured by benchmarks/state_variance_mixing.py.
        warnings.push(slow_state_variance_mixing(
            "the seasonal local-level Gibbs sampler's level and seasonal variances",
            1.011,
            1.595,
        ));
        warnings
    }

    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<SeasonalPosteriorPredictiveForecast, BayesianForecastError> {
        if horizon == 0 {
            return Err(invalid_config("forecast horizon must be positive"));
        }
        if self.period < 2 || self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(invalid_config(
                "posterior must contain a valid period and non-empty chains",
            ));
        }
        check_forecast_size("seasonal forecast", &self.chains, horizon, 4)?;
        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &SeasonalLocalLevelPosteriorDraw, rng| {
                validate_positive_variance("level", draw.level_variance)?;
                validate_positive_variance("seasonal", draw.seasonal_variance)?;
                validate_positive_variance("observation", draw.observation_variance)?;
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
    // Level, seasonal and observation variances.
    let observed_count = require_finite_observations(observations, 3, "seasonal local-level")?;
    // Each backward conditional holds a d x d gain and a d x 2d root.
    schedule.check_fit_size(
        "seasonal fit",
        &[config.period + 3],
        &[observations.len() + 1, config.period, config.period, 3],
    )?;
    let transitions = observations.len();
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
    .map_err(|error| numerical(format!("could not build seasonal state model: {error}")))?;
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
                .map_err(|error| numerical(format!("seasonal FFBS failed: {error}")))?;

            let mut level_sum_sq = 0.0;
            let mut seasonal_sum_sq = 0.0;
            for pair in states.windows(2) {
                let level_residual = pair[1][0] - pair[0][0];
                let seasonal_residual = pair[1][1] + pair[0][1..].iter().sum::<f64>();
                level_sum_sq += level_residual * level_residual;
                seasonal_sum_sq += seasonal_residual * seasonal_residual;
            }
            *level_variance =
                config
                    .level_variance_prior
                    .conditional_draw(transitions, level_sum_sq, rng)?;
            *seasonal_variance = config.seasonal_variance_prior.conditional_draw(
                transitions,
                seasonal_sum_sq,
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
            *observation_variance = config.observation_variance_prior.conditional_draw(
                observed_count,
                observation_sum_sq,
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
        // One finite observation per inferred variance: three.
        assert!(fit_bayesian_seasonal_local_level(&[0.0; 3], &config()).is_ok());
        assert!(matches!(
            fit_bayesian_seasonal_local_level(&[0.0, f64::NAN, 0.0, f64::NAN], &config()),
            Err(BayesianForecastError::InvalidObservations(_))
        ));
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

    /// The draws against moments of the exact joint posterior. Given the three
    /// variances, the model is the textbook dummy seasonal: the level is a
    /// random walk, the new seasonal effect is minus the sum of the previous
    /// `period - 1` plus an innovation, and each observation is level plus
    /// effect plus error. Written from that definition, without the
    /// state-space matrices or FFBS the sampler uses. The initial effects are
    /// placed as documented, one cycle in forecast order: their expected value
    /// at step `t` is `initial_seasonal_effects[t % period]`. The initial
    /// states' independence and variances follow `seasonal_local_level`, so
    /// this test does not check them.
    #[test]
    fn draws_match_the_exact_posterior_of_every_variance_and_the_terminal_state() {
        use crate::exact_reference::TermVariance::{Fixed, Inferred};
        use crate::exact_reference::{
            assert_matches, exact_posterior, seed_offset, Axis, Linear, LinearGaussianModel,
        };
        use rand::SeedableRng;

        let period = 4;
        let steps = 16;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1301);
        let mut level = 5.0;
        let mut effects = vec![1.2, -0.3, -0.9];
        let mut observations = Vec::with_capacity(steps);
        for _ in 0..steps {
            level += standard_normal(&mut rng) * 0.4;
            let effect = -effects.iter().sum::<f64>() + standard_normal(&mut rng) * 0.2;
            effects.insert(0, effect);
            effects.pop();
            observations.push(level + effect + standard_normal(&mut rng) * 0.5);
        }
        observations[5] = f64::NAN;
        // The three priors differ in shape as well as scale, so an update
        // that reads another variance's prior shows.
        let config = BayesianSeasonalLocalLevelConfig {
            seasonal_variance_prior: InverseGammaPrior::new(4.0, 0.15).unwrap(),
            observation_variance_prior: InverseGammaPrior::new(3.5, 0.6).unwrap(),
            num_chains: 4,
            num_warmup: 1000,
            num_draws: 25_000,
            seed: 7 + seed_offset(),
            ..config()
        };
        let initial = &config.initial_seasonal_effects;
        assert!(initial.iter().any(|&effect| effect != 0.0));

        // Terms: the initial level and `period - 1` initial effects, then one
        // level innovation, one seasonal innovation and one error per step.
        let mut terms = vec![Fixed(config.initial_level_variance)];
        terms.extend((1..period).map(|_| Fixed(config.initial_seasonal_variance)));
        terms.extend((0..steps).map(|_| Inferred(0)));
        terms.extend((0..steps).map(|_| Inferred(1)));
        terms.extend((0..steps).map(|_| Inferred(2)));
        let count = terms.len();
        let mut level = Linear {
            constant: config.initial_level,
            ..Linear::term(0, count)
        };
        // Newest first: effects[0] is the current effect. The effect `lag`
        // steps before the first observation belongs to step `-lag`, which is
        // step `period - lag` of the cycle.
        let mut effects = (1..period)
            .map(|lag| Linear {
                constant: initial[period - lag],
                ..Linear::term(lag, count)
            })
            .collect::<Vec<_>>();
        let mut observed = Vec::new();
        for (t, &value) in observations.iter().enumerate() {
            level = level.plus(&Linear::term(period + t, count));
            let effect = effects
                .iter()
                .fold(Linear::term(period + steps + t, count), |sum, previous| {
                    sum.scaled_plus(-1.0, previous)
                });
            assert!((effect.constant - initial[t % period]).abs() < 1e-12);
            effects.insert(0, effect);
            effects.pop();
            if value.is_finite() {
                let error = Linear::term(period + 2 * steps + t, count);
                observed.push((level.plus(&effects[0]).plus(&error), value));
            }
        }
        let model = LinearGaussianModel {
            terms,
            observed,
            latents: std::iter::once(level).chain(effects).collect(),
        };
        let priors = [
            config.level_variance_prior,
            config.seasonal_variance_prior,
            config.observation_variance_prior,
        ];
        let exact = exact_posterior(
            &model,
            &priors,
            &[
                Axis {
                    low: -6.0,
                    high: 2.0,
                    points: 55,
                },
                Axis {
                    low: -8.0,
                    high: 3.0,
                    points: 75,
                },
                Axis {
                    low: -6.0,
                    high: 3.0,
                    points: 61,
                },
            ],
        );

        let posterior = fit_bayesian_seasonal_local_level(&observations, &config).unwrap();
        let draws = |name: String, pick: &dyn Fn(&SeasonalLocalLevelPosteriorDraw) -> f64| {
            let chains = posterior
                .chains
                .iter()
                .map(|chain| chain.iter().map(pick).collect())
                .collect();
            (name, chains)
        };
        let mut all = vec![
            draws("level_variance".into(), &|d| d.level_variance),
            draws("seasonal_variance".into(), &|d| d.seasonal_variance),
            draws("observation_variance".into(), &|d| d.observation_variance),
        ];
        all.extend(
            (0..period).map(|i| draws(format!("terminal_state[{i}]"), &|d| d.terminal_state[i])),
        );
        assert_matches(&all, &exact, &priors);
    }
}
