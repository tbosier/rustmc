//! Joint Gaussian regression and structural-state Gibbs/FFBS inference.
//!
//! Proper Gaussian coefficient priors are independent of variance priors.
//! Coefficients are static latent states with exactly zero innovation variance;
//! every forecast path retains one joint coefficient/state/variance draw.
use crate::bayesian_forecast::InverseGammaPrior;
use crate::state_space::{LinearGaussianStateSpace, StateSpaceError};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct GaussianCoefficientPrior {
    pub mean: Vec<f64>,
    /// Full, positive-definite covariance in row-major order.
    pub covariance: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RegressionConfig {
    pub structural_model: LinearGaussianStateSpace,
    /// Coordinates with independent Gaussian state innovations.
    pub innovation_indices: Vec<usize>,
    pub variance_priors: Vec<InverseGammaPrior>,
    pub variance_names: Vec<String>,
    pub observation_variance_prior: InverseGammaPrior,
    pub coefficient_prior: GaussianCoefficientPrior,
    pub num_chains: usize,
    pub num_draws: usize,
    pub num_warmup: usize,
    pub thinning: usize,
    pub seed: u64,
    /// A structural seasonal observation uses coordinates zero and one.
    pub seasonal: bool,
}

impl RegressionConfig {
    pub fn from_local_level(
        c: &crate::bayesian_forecast::BayesianLocalLevelConfig,
        coefficient_prior: GaussianCoefficientPrior,
    ) -> Result<Self, StateSpaceError> {
        Ok(Self {
            structural_model: LinearGaussianStateSpace::local_level(
                1.0,
                1.0,
                c.initial_mean,
                c.initial_variance,
            )?,
            innovation_indices: vec![0],
            variance_priors: vec![c.process_variance_prior],
            variance_names: vec!["process_variance".into()],
            observation_variance_prior: c.observation_variance_prior,
            coefficient_prior,
            num_chains: c.num_chains,
            num_draws: c.num_draws,
            num_warmup: c.num_warmup,
            thinning: c.thinning,
            seed: c.seed,
            seasonal: false,
        })
    }
    pub fn from_trend(
        c: &crate::bayesian_trend::BayesianLocalLinearTrendConfig,
        coefficient_prior: GaussianCoefficientPrior,
    ) -> Result<Self, StateSpaceError> {
        Ok(Self {
            structural_model: LinearGaussianStateSpace::new(
                2,
                vec![1.0, 1.0, 0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0, 0.0, 1.0],
                1.0,
                c.initial_mean.to_vec(),
                c.initial_covariance.to_vec(),
            )?,
            innovation_indices: vec![0, 1],
            variance_priors: vec![c.level_variance_prior, c.slope_variance_prior],
            variance_names: vec!["level_variance".into(), "slope_variance".into()],
            observation_variance_prior: c.observation_variance_prior,
            coefficient_prior,
            num_chains: c.num_chains,
            num_draws: c.num_draws,
            num_warmup: c.num_warmup,
            thinning: c.thinning,
            seed: c.seed,
            seasonal: false,
        })
    }
    pub fn from_seasonal(
        c: &crate::bayesian_seasonal::BayesianSeasonalLocalLevelConfig,
        coefficient_prior: GaussianCoefficientPrior,
    ) -> Result<Self, StateSpaceError> {
        Ok(Self {
            structural_model: LinearGaussianStateSpace::seasonal_local_level(
                c.period,
                1.0,
                1.0,
                1.0,
                c.initial_level,
                c.initial_seasonal_effects.clone(),
                c.initial_level_variance,
                c.initial_seasonal_variance,
            )?,
            innovation_indices: vec![0, 1],
            variance_priors: vec![c.level_variance_prior, c.seasonal_variance_prior],
            variance_names: vec!["level_variance".into(), "seasonal_variance".into()],
            observation_variance_prior: c.observation_variance_prior,
            coefficient_prior,
            num_chains: c.num_chains,
            num_draws: c.num_draws,
            num_warmup: c.num_warmup,
            thinning: c.thinning,
            seed: c.seed,
            seasonal: true,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionDraw {
    pub variances: Vec<f64>,
    pub observation_variance: f64,
    pub coefficients: Vec<f64>,
    pub terminal_state: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RegressionPosterior {
    pub config: RegressionConfig,
    pub chains: Vec<Vec<RegressionDraw>>,
}

pub type Paths = Vec<Vec<Vec<f64>>>;
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegressionForecast {
    pub level_paths: Paths,
    pub secondary_paths: Paths,
    pub regression_paths: Paths,
    pub mean_paths: Paths,
    pub observation_paths: Paths,
    pub cumulative_observation_paths: Paths,
}

fn invalid(message: &str) -> StateSpaceError {
    StateSpaceError::InvalidParameter(message.into())
}
fn seed_for(seed: u64, chain: usize, domain: u64) -> u64 {
    let mut x = seed
        .wrapping_add(domain)
        .wrapping_add((chain as u64).wrapping_mul(0x9E3779B97F4A7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
fn draw_variance(
    prior: InverseGammaPrior,
    count: usize,
    ss: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, StateSpaceError> {
    let gamma = Gamma::new(
        prior.shape + count as f64 / 2.0,
        1.0 / (prior.scale + ss / 2.0),
    )
    .map_err(|_| invalid("invalid inverse-gamma update"))?;
    let v = 1.0 / gamma.sample(rng);
    if !v.is_finite() || v <= 0.0 {
        return Err(StateSpaceError::NumericalFailure(
            "sampled variance is not finite and positive".into(),
        ));
    }
    Ok(v)
}
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

pub fn fit_regression(
    y: &[f64],
    design: &[Vec<f64>],
    config: &RegressionConfig,
) -> Result<RegressionPosterior, StateSpaceError> {
    if y.iter().any(|x| x.is_infinite()) || y.iter().filter(|x| x.is_finite()).count() < 2 {
        return Err(invalid(
            "observations require at least two finite values and no infinities",
        ));
    }
    if design.len() != y.len() {
        return Err(invalid(
            "exog must have one row per observation, including missing observations",
        ));
    }
    if config.num_chains == 0 || config.num_draws == 0 || config.thinning == 0 {
        return Err(invalid("chains, draws and thinning must be positive"));
    }
    let iterations = config
        .num_draws
        .checked_mul(config.thinning)
        .and_then(|n| n.checked_add(config.num_warmup))
        .ok_or_else(|| invalid("too many iterations"))?;
    let d = config.structural_model.dimension();
    if config.innovation_indices.len() != config.variance_priors.len()
        || config.variance_names.len() != config.variance_priors.len()
        || config.innovation_indices.iter().any(|&i| i >= d)
    {
        return Err(invalid("innovation indices, priors and names must align"));
    }
    for prior in config
        .variance_priors
        .iter()
        .chain(std::iter::once(&config.observation_variance_prior))
    {
        if !prior.shape.is_finite()
            || prior.shape <= 0.0
            || !prior.scale.is_finite()
            || prior.scale <= 0.0
        {
            return Err(invalid(
                "variance priors require finite positive shape and scale",
            ));
        }
    }
    let template = config.structural_model.with_static_regression(
        design,
        &config.coefficient_prior.mean,
        &config.coefficient_prior.covariance,
    )?;
    let observed = y.iter().filter(|x| x.is_finite()).count();
    let chains = (0..config.num_chains)
        .into_par_iter()
        .map(|chain| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed_for(config.seed, chain, 0x5245475f464954));
            let mut model = template.clone();
            let mut variances: Vec<f64> = config
                .variance_priors
                .iter()
                .map(|p| p.scale / (p.shape + 1.0))
                .collect();
            let mut observation_variance = config.observation_variance_prior.scale
                / (config.observation_variance_prior.shape + 1.0);
            let mut draws = Vec::with_capacity(config.num_draws);
            for iteration in 0..iterations {
                model.set_variances(&config.innovation_indices, &variances, observation_variance);
                let states = model.sample_states_ffbs(y, &mut rng)?;
                for (variance, (&index, &prior)) in variances.iter_mut().zip(
                    config
                        .innovation_indices
                        .iter()
                        .zip(&config.variance_priors),
                ) {
                    let ss = states
                        .windows(2)
                        .map(|pair| {
                            let predicted: f64 = config.structural_model.transition()
                                [index * d..(index + 1) * d]
                                .iter()
                                .zip(&pair[0][..d])
                                .map(|(a, b)| a * b)
                                .sum();
                            (pair[1][index] - predicted).powi(2)
                        })
                        .sum();
                    *variance = draw_variance(prior, y.len(), ss, &mut rng)?;
                }
                let ss = y
                    .iter()
                    .zip(design)
                    .zip(&states[1..])
                    .filter(|((y, _), _)| y.is_finite())
                    .map(|((&y, x), state)| {
                        let structural: f64 = config
                            .structural_model
                            .observation()
                            .iter()
                            .zip(state)
                            .map(|(a, b)| a * b)
                            .sum();
                        let regression: f64 = x.iter().zip(&state[d..]).map(|(a, b)| a * b).sum();
                        (y - structural - regression).powi(2)
                    })
                    .sum();
                observation_variance =
                    draw_variance(config.observation_variance_prior, observed, ss, &mut rng)?;
                if iteration >= config.num_warmup
                    && (iteration + 1 - config.num_warmup).is_multiple_of(config.thinning)
                {
                    let state = states.last().expect("validated nonempty data");
                    draws.push(RegressionDraw {
                        variances: variances.clone(),
                        observation_variance,
                        coefficients: state[d..].to_vec(),
                        terminal_state: state[..d].to_vec(),
                    });
                }
            }
            Ok(draws)
        })
        .collect::<Result<Vec<_>, StateSpaceError>>()?;
    Ok(RegressionPosterior {
        config: config.clone(),
        chains,
    })
}

impl RegressionPosterior {
    pub fn forecast(
        &self,
        design: &[Vec<f64>],
        seed: u64,
    ) -> Result<RegressionForecast, StateSpaceError> {
        let p = self.config.coefficient_prior.mean.len();
        if design.is_empty()
            || design
                .iter()
                .any(|row| row.len() != p || row.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "future exog must be nonempty, finite, and match training feature count/order",
            ));
        }
        let d = self.config.structural_model.dimension();
        if self.chains.is_empty()
            || self.chains.iter().any(Vec::is_empty)
            || self
                .config
                .innovation_indices
                .iter()
                .any(|&index| index >= d)
            || self.chains.iter().flatten().any(|draw| {
                draw.coefficients.len() != p
                    || draw.terminal_state.len() != d
                    || draw.variances.len() != self.config.innovation_indices.len()
                    || draw
                        .coefficients
                        .iter()
                        .chain(&draw.terminal_state)
                        .any(|v| !v.is_finite())
                    || draw
                        .variances
                        .iter()
                        .chain(std::iter::once(&draw.observation_variance))
                        .any(|v| !v.is_finite() || *v <= 0.0)
            })
        {
            return Err(invalid(
                "posterior chains must contain finite, dimensionally valid joint draws",
            ));
        }
        let chains = self
            .chains
            .par_iter()
            .enumerate()
            .map(|(chain, draws)| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed_for(seed, chain, 0x5245475f50524544));
                let mut result = RegressionForecast::default();
                let (
                    mut levels,
                    mut secondary,
                    mut regression,
                    mut means,
                    mut observations,
                    mut cumulative,
                ) = (vec![], vec![], vec![], vec![], vec![], vec![]);
                for draw in draws {
                    let mut state = draw.terminal_state.clone();
                    let (mut l, mut s, mut r, mut m, mut o, mut c) =
                        (vec![], vec![], vec![], vec![], vec![], vec![]);
                    let mut total = 0.0;
                    for row in design {
                        let mut next: Vec<f64> = self
                            .config
                            .structural_model
                            .transition()
                            .chunks(d)
                            .map(|a| a.iter().zip(&state).map(|(a, b)| a * b).sum())
                            .collect();
                        for (&i, &v) in self.config.innovation_indices.iter().zip(&draw.variances) {
                            next[i] += normal(&mut rng) * v.sqrt();
                        }
                        let reg: f64 = row.iter().zip(&draw.coefficients).map(|(a, b)| a * b).sum();
                        let mean = reg
                            + self
                                .config
                                .structural_model
                                .observation()
                                .iter()
                                .zip(&next)
                                .map(|(a, b)| a * b)
                                .sum::<f64>();
                        let observation =
                            mean + normal(&mut rng) * draw.observation_variance.sqrt();
                        total += observation;
                        if !total.is_finite() || next.iter().any(|v| !v.is_finite()) {
                            return Err(StateSpaceError::NumericalFailure(
                                "forecast overflowed".into(),
                            ));
                        }
                        l.push(next[0]);
                        s.push(if d > 1 { next[1] } else { 0.0 });
                        r.push(reg);
                        m.push(mean);
                        o.push(observation);
                        c.push(total);
                        state = next;
                    }
                    levels.push(l);
                    secondary.push(s);
                    regression.push(r);
                    means.push(m);
                    observations.push(o);
                    cumulative.push(c);
                }
                result.level_paths.push(levels);
                result.secondary_paths.push(secondary);
                result.regression_paths.push(regression);
                result.mean_paths.push(means);
                result.observation_paths.push(observations);
                result.cumulative_observation_paths.push(cumulative);
                Ok(result)
            })
            .collect::<Result<Vec<_>, StateSpaceError>>()?;
        let mut result = RegressionForecast::default();
        for c in chains {
            result.level_paths.extend(c.level_paths);
            result.secondary_paths.extend(c.secondary_paths);
            result.regression_paths.extend(c.regression_paths);
            result.mean_paths.extend(c.mean_paths);
            result.observation_paths.extend(c.observation_paths);
            result
                .cumulative_observation_paths
                .extend(c.cumulative_observation_paths);
        }
        Ok(result)
    }
}

/// Calendar Fourier rows at integer positions `start..start+count`. Column order
/// is sin(1), cos(1), ..., with only cos at the even-period Nyquist harmonic.
pub fn fourier_design(
    count: usize,
    period: usize,
    harmonics: usize,
    start: i64,
) -> Result<Vec<Vec<f64>>, StateSpaceError> {
    if period < 2 || harmonics == 0 || harmonics > period / 2 {
        return Err(invalid(
            "period must be at least two; harmonics must be between one and floor(period/2)",
        ));
    }
    let mut rows = Vec::with_capacity(count);
    for offset in 0..count {
        let time = i128::from(start) + offset as i128;
        let phase = time.rem_euclid(period as i128) as f64;
        let mut row = Vec::with_capacity(2 * harmonics);
        for k in 1..=harmonics {
            let angle = std::f64::consts::TAU * k as f64 * phase / period as f64;
            if 2 * k != period {
                row.push(angle.sin());
            }
            row.push(angle.cos());
        }
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_static_regression_matches_analytic_posterior_and_future_covariance() {
        let y = vec![1.0, 2.0, f64::NAN, 4.0];
        let x = vec![vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
        let model = LinearGaussianStateSpace::local_level(0.0, 0.5, 0.0, 2.0)
            .unwrap()
            .with_static_regression(&x, &[0.0], &[3.0])
            .unwrap();
        // Posterior precision: diag(1/2,1/3) + Z'Z / .5, omitting missing y.
        let (a, b, c) = (6.5, 2.0, 10.0 + 1.0 / 3.0);
        let det = a * c - b * b;
        let covariance = [c / det, -b / det, -b / det, a / det];
        let mean = [
            covariance[0] * 14.0 + covariance[1] * 14.0,
            covariance[2] * 14.0 + covariance[3] * 14.0,
        ];
        let smooth = model.smooth(&y).unwrap();
        for m in &smooth.smoothed_means {
            for i in 0..2 {
                assert!((m[i] - mean[i]).abs() < 1e-10);
            }
        }
        for v in &smooth.smoothed_covariances {
            for i in 0..4 {
                assert!((v[i] - covariance[i]).abs() < 1e-10);
            }
        }
        let future = vec![vec![1.0, 3.0], vec![1.0, -2.0]];
        let forecast = model.forecast_with_observation_rows(&y, &future).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = covariance[0]
                    + future[i][1] * covariance[2]
                    + future[j][1] * covariance[1]
                    + future[i][1] * future[j][1] * covariance[3]
                    + if i == j { 0.5 } else { 0.0 };
                assert!((forecast.observation_covariance[2 * i + j] - expected).abs() < 1e-10);
            }
        }
        assert!(
            (forecast.cumulative_observation_variances[1]
                - forecast.observation_covariance.iter().sum::<f64>())
            .abs()
                < 1e-10
        );
        let mut rng = ChaCha8Rng::seed_from_u64(182);
        let mut draws = vec![];
        for _ in 0..6000 {
            let states = model.sample_states_ffbs(&y, &mut rng).unwrap();
            for state in &states {
                for (i, value) in state.iter().enumerate() {
                    assert!((value - states[0][i]).abs() < 1e-6);
                }
            }
            draws.push(states[0].clone());
        }
        let empirical: Vec<f64> = (0..2)
            .map(|i| draws.iter().map(|d| d[i]).sum::<f64>() / draws.len() as f64)
            .collect();
        for i in 0..2 {
            assert!((empirical[i] - mean[i]).abs() < 0.025);
            for j in 0..2 {
                let v = draws
                    .iter()
                    .map(|d| (d[i] - empirical[i]) * (d[j] - empirical[j]))
                    .sum::<f64>()
                    / draws.len() as f64;
                assert!((v - covariance[2 * i + j]).abs() < 0.012);
            }
        }
    }
    fn config() -> RegressionConfig {
        RegressionConfig {
            structural_model: LinearGaussianStateSpace::local_level(1.0, 1.0, 0.0, 2.0).unwrap(),
            innovation_indices: vec![0],
            variance_priors: vec![InverseGammaPrior {
                shape: 3.0,
                scale: 0.08,
            }],
            variance_names: vec!["process_variance".into()],
            observation_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 0.4,
            },
            coefficient_prior: GaussianCoefficientPrior {
                mean: vec![0.0],
                covariance: vec![9.0],
            },
            num_chains: 2,
            num_draws: 180,
            num_warmup: 120,
            thinning: 1,
            seed: 32,
            seasonal: false,
        }
    }
    #[test]
    fn seeded_recovery_and_pool_independence() {
        let mut rng = ChaCha8Rng::seed_from_u64(188);
        let mut level = 0.0;
        let x: Vec<Vec<f64>> = (0..100).map(|_| vec![normal(&mut rng)]).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|x| {
                level += normal(&mut rng) * 0.02_f64.sqrt();
                level + 1.8 * x[0] + normal(&mut rng) * 0.1_f64.sqrt()
            })
            .collect();
        let fit = fit_regression(&y, &x, &config()).unwrap();
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| fit_regression(&y, &x, &config()))
            .unwrap();
        assert_eq!(fit.chains, single.chains);
        let coefficients: Vec<f64> = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.coefficients[0])
            .collect();
        let mean = coefficients.iter().sum::<f64>() / coefficients.len() as f64;
        assert!((mean - 1.8).abs() < 0.12, "{mean}");
        let obs = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.observation_variance)
            .sum::<f64>()
            / coefficients.len() as f64;
        assert!((0.05..0.2).contains(&obs), "{obs}");
        let future = vec![vec![1.0]; 4];
        let paths = fit.forecast(&future, 7).unwrap();
        assert_eq!(paths, fit.forecast(&future, 7).unwrap());
        for ((reg, means), levels) in paths
            .regression_paths
            .iter()
            .flatten()
            .zip(paths.mean_paths.iter().flatten())
            .zip(paths.level_paths.iter().flatten())
        {
            assert!(reg.iter().all(|v| *v == reg[0]));
            for h in 0..4 {
                assert!((means[h] - reg[h] - levels[h]).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn fourier_phase_and_nyquist() {
        let all = fourier_design(30, 12, 6, 0).unwrap();
        assert_eq!(all[0].len(), 11);
        assert_eq!(fourier_design(12, 12, 6, 18).unwrap(), all[18..]);
        assert_eq!(all[0], all[12]);
        assert!(fourier_design(1, 12, 7, 0).is_err());
        assert!(fourier_design(1, 1, 1, 0).is_err());
        assert_eq!(
            fourier_design(1, 12, 2, -1).unwrap(),
            fourier_design(1, 12, 2, 11).unwrap()
        );
    }
}
