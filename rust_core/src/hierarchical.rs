//! Joint partial pooling for ragged collections of Gaussian series.
//!
//! The model is a three-tier normal hierarchy:
//!
//! ```text
//! population_mean ~ Normal(prior_mean, prior_variance)
//! group_mean[g] ~ Normal(population_mean, group_variance)
//! program_mean[p] ~ Normal(group_mean[group[p]], program_variance)
//! y[p, t] ~ Normal(program_mean[p], observation_variance)
//! ```
//!
//! All three variances have inverse-gamma priors. A conjugate Gibbs
//! sampler draws every full conditional directly, so this specialized model
//! does not require Hamiltonian trajectories through the funnel geometry of a
//! centered hierarchical parameterization. Series may have different lengths;
//! `NaN` values are retained as missing positions and ignored by the likelihood.

use crate::bayesian_forecast::{BayesianForecastError, InverseGammaPrior};
use crate::diagnostics::DiagnosticsReport;
use crate::forecast_common::{
    checked_value_count, overdispersed_location, overdispersed_positive, run_gibbs_chains,
    sample_inverse_gamma, simulate_draws, split_paths, GibbsSchedule, MAX_MATERIALIZED_VALUES,
};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Retained posterior values may reach twice the forecast limit: a
/// hierarchy with many programs is the case this model exists for.
const MAX_POSTERIOR_VALUES: usize = 2 * MAX_MATERIALIZED_VALUES;

/// Configuration for a joint hierarchical-normal fit.
#[derive(Debug, Clone)]
pub struct HierarchicalMeanConfig {
    pub population_mean_prior: f64,
    pub population_variance_prior: f64,
    pub group_variance_prior: InverseGammaPrior,
    pub program_variance_prior: InverseGammaPrior,
    pub observation_variance_prior: InverseGammaPrior,
    pub num_chains: usize,
    pub num_warmup: usize,
    pub num_draws: usize,
    pub thinning: usize,
    pub seed: u64,
}

impl HierarchicalMeanConfig {
    fn validate(&self) -> Result<GibbsSchedule, BayesianForecastError> {
        if !self.population_mean_prior.is_finite() {
            return Err(invalid_config("population mean prior must be finite"));
        }
        if !self.population_variance_prior.is_finite() || self.population_variance_prior <= 0.0 {
            return Err(invalid_config(
                "population variance prior must be finite and strictly positive",
            ));
        }
        self.group_variance_prior.validate("group-variance prior")?;
        self.program_variance_prior
            .validate("program-variance prior")?;
        self.observation_variance_prior
            .validate("observation-variance prior")?;
        GibbsSchedule::new(
            self.num_chains,
            self.num_warmup,
            self.num_draws,
            self.thinning,
        )
    }
}

/// One coherent posterior draw from the complete hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalMeanPosteriorDraw {
    pub population_mean: f64,
    pub group_variance: f64,
    pub program_variance: f64,
    pub observation_variance: f64,
    pub group_means: Vec<f64>,
    pub program_means: Vec<f64>,
}

/// Joint posterior samples and immutable hierarchy metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalMeanPosterior {
    /// Samples indexed as `[chain][draw]`.
    pub chains: Vec<Vec<HierarchicalMeanPosteriorDraw>>,
    /// For each program, the zero-based group containing it.
    pub group_index: Vec<usize>,
    /// Number of finite observations supplied for each program.
    pub observed_counts: Vec<usize>,
    pub group_count: usize,
}

/// Coherent posterior-predictive paths for every program.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalMeanForecast {
    /// Latent expected levels indexed `[chain][draw][program]`. The static
    /// level is repeated across forecast steps only when materialized by a
    /// caller, avoiding a redundant core allocation.
    pub state_means: Vec<Vec<Vec<f64>>>,
    /// Future observations indexed `[chain][draw][program * horizon + step]`.
    /// Each draw is contiguous to avoid one allocation per program path.
    pub observation_paths: Vec<Vec<Vec<f64>>>,
    pub group_index: Vec<usize>,
    pub group_count: usize,
    pub horizon: usize,
}

impl HierarchicalMeanPosterior {
    pub fn program_count(&self) -> usize {
        self.group_index.len()
    }

    /// Compute rank-normalized R-hat, bulk/tail ESS, MCSE, and HDIs for every
    /// population, variance, group, and program parameter.
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let mut names = vec![
            "population_mean".to_string(),
            "group_variance".to_string(),
            "program_variance".to_string(),
            "observation_variance".to_string(),
        ];
        names.extend((0..self.group_count).map(|index| format!("group_mean[{index}]")));
        names.extend((0..self.program_count()).map(|index| format!("program_mean[{index}]")));
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|draw| {
                        let mut values = Vec::with_capacity(names.len());
                        values.extend([
                            draw.population_mean,
                            draw.group_variance,
                            draw.program_variance,
                            draw.observation_variance,
                        ]);
                        values.extend_from_slice(&draw.group_means);
                        values.extend_from_slice(&draw.program_means);
                        values
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        crate::forecast_diagnostics::parameter_diagnostics(&samples, &names)
    }

    /// Generate aligned future paths. Summing the program axis within a draw
    /// preserves posterior dependence induced by the shared hierarchy.
    pub fn forecast(
        &self,
        horizon: usize,
        seed: u64,
    ) -> Result<HierarchicalMeanForecast, BayesianForecastError> {
        if horizon == 0 {
            return Err(invalid_config("forecast horizon must be positive"));
        }
        if self.chains.is_empty() || self.chains.iter().any(Vec::is_empty) {
            return Err(invalid_config(
                "every posterior chain must contain at least one draw",
            ));
        }
        let program_count = self.program_count();
        let chain_count = self.chains.len();
        let draw_count = self.chains.first().map_or(0, Vec::len);
        if self.chains.iter().any(|chain| chain.len() != draw_count) {
            return Err(invalid_config(
                "every posterior chain must contain the same number of draws",
            ));
        }
        checked_value_count(
            "hierarchical forecast",
            &[chain_count, draw_count, program_count, horizon],
            MAX_MATERIALIZED_VALUES,
        )?;
        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &HierarchicalMeanPosteriorDraw, rng| {
                if draw.program_means.len() != program_count {
                    return Err(numerical("posterior program dimension is inconsistent"));
                }
                validate_variance(draw.observation_variance, "observation")?;
                let observation_sd = draw.observation_variance.sqrt();
                let draw_len = program_count * horizon;
                let mut draw_observations = Vec::new();
                draw_observations
                    .try_reserve_exact(draw_len)
                    .map_err(|error| {
                        invalid_config(format!(
                            "could not reserve posterior predictive storage: {error}"
                        ))
                    })?;
                for &program_mean in &draw.program_means {
                    if !program_mean.is_finite() {
                        return Err(numerical("posterior program mean is not finite"));
                    }
                    for _ in 0..horizon {
                        let observation = program_mean + standard_normal(rng) * observation_sd;
                        if !observation.is_finite() {
                            return Err(numerical("posterior predictive simulation overflowed"));
                        }
                        draw_observations.push(observation);
                    }
                }
                Ok([draw.program_means.clone(), draw_observations])
            },
        )?;
        let [state_means, observation_paths] = split_paths(per_draw);
        Ok(HierarchicalMeanForecast {
            state_means,
            observation_paths,
            group_index: self.group_index.clone(),
            group_count: self.group_count,
            horizon,
        })
    }
}

impl HierarchicalMeanForecast {
    pub fn chain_count(&self) -> usize {
        self.observation_paths.len()
    }

    pub fn draw_count(&self) -> usize {
        self.observation_paths.first().map_or(0, Vec::len)
    }

    pub fn program_count(&self) -> usize {
        self.group_index.len()
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }
}

/// Fit every program jointly under one shared posterior.
pub fn fit_hierarchical_mean(
    series: &[Vec<f64>],
    group_index: &[usize],
    config: &HierarchicalMeanConfig,
) -> Result<HierarchicalMeanPosterior, BayesianForecastError> {
    let schedule = config.validate()?;
    let validated = validate_inputs(series, group_index)?;
    let program_count = series.len();
    let group_count = validated.group_count;
    let total_observed = validated.observed_counts.iter().sum::<usize>();
    let parameter_count = 4usize
        .checked_add(group_count)
        .and_then(|value| value.checked_add(program_count))
        .ok_or_else(|| invalid_config("posterior dimensions overflow"))?;
    checked_value_count(
        "hierarchical posterior",
        &[config.num_chains, config.num_draws, parameter_count],
        MAX_POSTERIOR_VALUES,
    )?;

    let chains = run_gibbs_chains(
        &schedule,
        config.seed,
        FIT_SEED_DOMAIN,
        |rng| {
            // Variances start around their prior modes, and every mean
            // around its data average by up to two of the starting
            // between-group or between-program standard deviations, so that
            // chains begin apart in every coordinate.
            let group_variance = overdispersed_positive(config.group_variance_prior.mode(), rng);
            let program_variance =
                overdispersed_positive(config.program_variance_prior.mode(), rng);
            let observation_variance =
                overdispersed_positive(config.observation_variance_prior.mode(), rng);
            let group_means = validated
                .group_members
                .iter()
                .map(|members| {
                    let group_sum = members
                        .iter()
                        .map(|&program| validated.observation_sums[program])
                        .sum::<f64>();
                    let group_observed = members
                        .iter()
                        .map(|&program| validated.observed_counts[program])
                        .sum::<usize>();
                    overdispersed_location(
                        group_sum / group_observed as f64,
                        group_variance.sqrt(),
                        rng,
                    )
                })
                .collect::<Vec<_>>();
            let program_means = validated
                .observation_sums
                .iter()
                .zip(&validated.observed_counts)
                .map(|(&sum, &count)| {
                    overdispersed_location(sum / count as f64, program_variance.sqrt(), rng)
                })
                .collect::<Vec<_>>();
            Ok::<_, BayesianForecastError>(ChainState {
                group_means,
                program_means,
                group_variance,
                program_variance,
                observation_variance,
            })
        },
        |state, rng, retain| {
            let ChainState {
                group_means,
                program_means,
                group_variance,
                program_variance,
                observation_variance,
            } = state;
            let population_mean = sample_normal_precision(
                config.population_mean_prior / config.population_variance_prior
                    + group_means.iter().sum::<f64>() / *group_variance,
                1.0 / config.population_variance_prior + group_count as f64 / *group_variance,
                rng,
            )?;

            for (group, group_mean) in group_means.iter_mut().enumerate() {
                let members = &validated.group_members[group];
                let member_sum = members
                    .iter()
                    .map(|&program| program_means[program])
                    .sum::<f64>();
                let member_count = members.len();
                *group_mean = sample_normal_precision(
                    population_mean / *group_variance + member_sum / *program_variance,
                    1.0 / *group_variance + member_count as f64 / *program_variance,
                    rng,
                )?;
            }

            for (program, program_mean) in program_means.iter_mut().enumerate() {
                let finite_sum = validated.observation_sums[program];
                let count = validated.observed_counts[program];
                *program_mean = sample_normal_precision(
                    group_means[group_index[program]] / *program_variance
                        + finite_sum / *observation_variance,
                    1.0 / *program_variance + count as f64 / *observation_variance,
                    rng,
                )?;
            }

            let observation_ss = program_means
                .iter()
                .enumerate()
                .map(|(program, &mean)| {
                    let sample_mean = validated.observation_sums[program]
                        / validated.observed_counts[program] as f64;
                    validated.within_program_sum_squares[program]
                        + validated.observed_counts[program] as f64 * (sample_mean - mean).powi(2)
                })
                .sum::<f64>();
            *observation_variance = sample_inverse_gamma(
                config.observation_variance_prior.shape + total_observed as f64 / 2.0,
                config.observation_variance_prior.scale + observation_ss / 2.0,
                rng,
            )?;

            let program_ss = program_means
                .iter()
                .zip(group_index)
                .map(|(&mean, &group)| (mean - group_means[group]).powi(2))
                .sum::<f64>();
            *program_variance = sample_inverse_gamma(
                config.program_variance_prior.shape + program_count as f64 / 2.0,
                config.program_variance_prior.scale + program_ss / 2.0,
                rng,
            )?;

            let group_ss = group_means
                .iter()
                .map(|&mean| (mean - population_mean).powi(2))
                .sum::<f64>();
            *group_variance = sample_inverse_gamma(
                config.group_variance_prior.shape + group_count as f64 / 2.0,
                config.group_variance_prior.scale + group_ss / 2.0,
                rng,
            )?;

            Ok(retain.then(|| HierarchicalMeanPosteriorDraw {
                population_mean,
                group_variance: *group_variance,
                program_variance: *program_variance,
                observation_variance: *observation_variance,
                group_means: group_means.clone(),
                program_means: program_means.clone(),
            }))
        },
    )?;

    Ok(HierarchicalMeanPosterior {
        chains,
        group_index: group_index.to_vec(),
        observed_counts: validated.observed_counts,
        group_count,
    })
}

/// The Gibbs state carried between sweeps; the population mean is drawn
/// first in every sweep, so it is not part of it.
struct ChainState {
    group_means: Vec<f64>,
    program_means: Vec<f64>,
    group_variance: f64,
    program_variance: f64,
    observation_variance: f64,
}

struct ValidatedInputs {
    observed_counts: Vec<usize>,
    observation_sums: Vec<f64>,
    within_program_sum_squares: Vec<f64>,
    group_members: Vec<Vec<usize>>,
    group_count: usize,
}

fn validate_inputs(
    series: &[Vec<f64>],
    group_index: &[usize],
) -> Result<ValidatedInputs, BayesianForecastError> {
    if series.is_empty() {
        return Err(invalid_observations("at least one program is required"));
    }
    if series.len() != group_index.len() {
        return Err(invalid_observations(
            "series and group_index must have the same length",
        ));
    }
    if series.iter().any(Vec::is_empty) {
        return Err(invalid_observations(
            "every program series must contain at least one position",
        ));
    }
    if series.iter().flatten().any(|value| value.is_infinite()) {
        return Err(invalid_observations(
            "observations may be finite or NaN, but not infinite",
        ));
    }
    let observed_counts = series
        .iter()
        .map(|values| values.iter().filter(|value| value.is_finite()).count())
        .collect::<Vec<_>>();
    if observed_counts.contains(&0) {
        return Err(invalid_observations(
            "every program must contain at least one finite observation",
        ));
    }
    let observation_sums = series
        .iter()
        .map(|values| {
            values
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let within_program_sum_squares = series
        .iter()
        .zip(&observation_sums)
        .zip(&observed_counts)
        .map(|((values, &sum), &count)| {
            let mean = sum / count as f64;
            values
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let max_group = group_index
        .iter()
        .copied()
        .max()
        .ok_or_else(|| invalid_observations("at least one group is required"))?;
    if max_group >= series.len() {
        return Err(invalid_observations(
            "group indices must be contiguous from zero with no empty groups",
        ));
    }
    let group_count = max_group
        .checked_add(1)
        .ok_or_else(|| invalid_observations("group index overflow"))?;
    let mut group_members = vec![Vec::new(); group_count];
    for (program, &group) in group_index.iter().enumerate() {
        group_members[group].push(program);
    }
    if group_members.iter().any(Vec::is_empty) {
        return Err(invalid_observations(
            "group indices must be contiguous from zero with no empty groups",
        ));
    }
    Ok(ValidatedInputs {
        observed_counts,
        observation_sums,
        within_program_sum_squares,
        group_members,
        group_count,
    })
}

fn sample_normal_precision(
    weighted_sum: f64,
    precision: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, BayesianForecastError> {
    let (mean, variance) = normal_moments_from_precision(weighted_sum, precision)?;
    let value = mean + standard_normal(rng) * variance.sqrt();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(numerical("normal simulation produced a non-finite draw"))
    }
}

fn normal_moments_from_precision(
    weighted_sum: f64,
    precision: f64,
) -> Result<(f64, f64), BayesianForecastError> {
    if !weighted_sum.is_finite() || !precision.is_finite() || precision <= 0.0 {
        return Err(numerical(
            "normal weighted sum and precision must be finite, with positive precision",
        ));
    }
    let variance = 1.0 / precision;
    let mean = weighted_sum * variance;
    if mean.is_finite() && variance.is_finite() && variance > 0.0 {
        Ok((mean, variance))
    } else {
        Err(numerical("normal conditional moments are not finite"))
    }
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

fn validate_variance(value: f64, name: &str) -> Result<(), BayesianForecastError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(numerical(format!(
            "{name} variance must be finite and strictly positive"
        )))
    }
}

fn invalid_config(message: impl Into<String>) -> BayesianForecastError {
    BayesianForecastError::InvalidConfiguration(message.into())
}

fn invalid_observations(message: impl Into<String>) -> BayesianForecastError {
    BayesianForecastError::InvalidObservations(message.into())
}

fn numerical(message: impl Into<String>) -> BayesianForecastError {
    BayesianForecastError::NumericalFailure(message.into())
}

const FIT_SEED_DOMAIN: u64 = 0x4649_545F_4849_4552;
const FORECAST_SEED_DOMAIN: u64 = 0x4652_4353_5F48_4945;

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> HierarchicalMeanConfig {
        HierarchicalMeanConfig {
            population_mean_prior: 0.0,
            population_variance_prior: 100.0,
            group_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 2.0,
            },
            program_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 2.0,
            },
            observation_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 2.0,
            },
            num_chains: 2,
            num_warmup: 80,
            num_draws: 120,
            thinning: 1,
            seed,
        }
    }

    #[test]
    fn ragged_joint_fit_and_forecast_are_seeded() {
        let series = vec![
            vec![0.0],
            vec![0.9, 1.1, 1.0],
            vec![9.0, 10.0, 11.0, f64::NAN],
        ];
        let groups = vec![0, 0, 1];
        let first = fit_hierarchical_mean(&series, &groups, &config(12)).unwrap();
        let second = fit_hierarchical_mean(&series, &groups, &config(12)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.observed_counts, vec![1, 3, 3]);
        assert_eq!(first.group_count, 2);
        let diagnostics = first.diagnostics();
        assert_eq!(diagnostics.params.len(), 9);
        assert_eq!(diagnostics.params[2].name, "program_variance");
        let forecast = first.forecast(4, 13).unwrap();
        assert_eq!(forecast, second.forecast(4, 13).unwrap());
        assert_eq!(forecast.chain_count(), 2);
        assert_eq!(forecast.draw_count(), 120);
        assert_eq!(forecast.program_count(), 3);
        assert_eq!(forecast.horizon(), 4);
        assert!(first.forecast(100_000, 13).is_err());
    }

    #[test]
    fn fit_and_forecast_are_bitwise_identical_across_rayon_pool_sizes() {
        let series = vec![
            vec![0.0],
            vec![0.9, 1.1, 1.0],
            vec![9.0, 10.0, 11.0, f64::NAN],
        ];
        let groups = vec![0, 0, 1];
        let run = || {
            let posterior = fit_hierarchical_mean(&series, &groups, &config(44)).unwrap();
            let forecast = posterior.forecast(4, 45).unwrap();
            (posterior, forecast)
        };
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        let multi_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(run);
        assert_eq!(single_thread, multi_thread);
    }

    #[test]
    fn one_point_program_is_shrunk_more_than_long_program() {
        let series = vec![vec![20.0], vec![10.0; 30], vec![10.0; 30]];
        let posterior = fit_hierarchical_mean(&series, &[0, 0, 0], &config(31)).unwrap();
        let one_point_mean = posterior
            .chains
            .iter()
            .flatten()
            .map(|draw| draw.program_means[0])
            .sum::<f64>()
            / 240.0;
        let long_mean = posterior
            .chains
            .iter()
            .flatten()
            .map(|draw| draw.program_means[1])
            .sum::<f64>()
            / 240.0;
        assert!(one_point_mean > 10.0 && one_point_mean < 20.0);
        assert!((20.0 - one_point_mean).abs() > (10.0 - long_mean).abs());
    }

    #[test]
    fn program_conditional_moments_match_analytic_adaptive_shrinkage() {
        let group_mean = 10.0;
        let sample_mean = 20.0;
        let program_variance = 4.0;
        let observation_variance = 9.0;
        for count in [1usize, 30] {
            let weighted_sum =
                group_mean / program_variance + count as f64 * sample_mean / observation_variance;
            let precision = 1.0 / program_variance + count as f64 / observation_variance;
            let (mean, variance) = normal_moments_from_precision(weighted_sum, precision).unwrap();
            let data_weight = count as f64 * program_variance
                / (count as f64 * program_variance + observation_variance);
            let expected_mean = data_weight * sample_mean + (1.0 - data_weight) * group_mean;
            assert!((mean - expected_mean).abs() < 1e-12);
            assert!((variance - 1.0 / precision).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_empty_programs_and_sparse_group_indices() {
        assert!(fit_hierarchical_mean(&[], &[], &config(1)).is_err());
        assert!(fit_hierarchical_mean(&[vec![]], &[0], &config(1)).is_err());
        assert!(fit_hierarchical_mean(&[vec![1.0]], &[1], &config(1)).is_err());
        assert!(fit_hierarchical_mean(&[vec![1.0]], &[usize::MAX - 1], &config(1)).is_err());
        assert!(fit_hierarchical_mean(&[vec![f64::NAN]], &[0], &config(1)).is_err());
    }
}
