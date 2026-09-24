//! Joint Gaussian regression and structural-state Gibbs/FFBS inference.
//!
//! Proper Gaussian coefficient priors are independent of variance priors.
//! Coefficients are static latent states with exactly zero innovation variance;
//! every forecast path retains one joint coefficient/state/variance draw.
use crate::bayesian_forecast::InverseGammaPrior;
use crate::forecast_common::{
    check_forecast_size, overdispersed_positive, require_finite_observations, run_gibbs_chains,
    sample_inverse_gamma, simulate_draws, split_paths, GibbsSchedule,
};
use crate::state_space::{LinearGaussianStateSpace, StateSpaceError};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// `"\0REG_FIT"`.
const FIT_SEED_DOMAIN: u64 = 0x0052_4547_5F46_4954;
/// `"REG_PRED"`.
const FORECAST_SEED_DOMAIN: u64 = 0x5245_475F_5052_4544;

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
fn draw_variance(
    prior: InverseGammaPrior,
    count: usize,
    ss: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, StateSpaceError> {
    Ok(sample_inverse_gamma(
        prior.shape + count as f64 / 2.0,
        prior.scale + ss / 2.0,
        rng,
    )?)
}
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

pub fn fit_regression(
    y: &[f64],
    design: &[Vec<f64>],
    config: &RegressionConfig,
) -> Result<RegressionPosterior, StateSpaceError> {
    if y.iter().any(|x| x.is_infinite()) {
        return Err(invalid(
            "observations may be finite or NaN, but not infinite",
        ));
    }
    // One finite observation per inferred variance, the observation variance
    // included.
    let observed =
        require_finite_observations(y, config.innovation_indices.len() + 1, "regression")?;
    if design.len() != y.len() {
        return Err(invalid(
            "exog must have one row per observation, including missing observations",
        ));
    }
    let schedule = GibbsSchedule::new(
        config.num_chains,
        config.num_warmup,
        config.num_draws,
        config.thinning,
    )?;
    if config.structural_model.has_observation_variances() {
        return Err(invalid(
            "regression variance inference requires a constant observation variance",
        ));
    }
    let d = config.structural_model.dimension();
    if config.innovation_indices.len() != config.variance_priors.len()
        || config.variance_names.len() != config.variance_priors.len()
        || config.innovation_indices.iter().any(|&i| i >= d)
    {
        return Err(invalid("innovation indices, priors and names must align"));
    }
    let mut unique = std::collections::HashSet::new();
    for &i in &config.innovation_indices {
        if !unique.insert(i)
            || (0..d)
                .any(|j| j != i && config.structural_model.process_covariance()[i * d + j] != 0.0)
        {
            return Err(invalid("inferred innovation indices must be unique and uncorrelated with all other innovations for inverse-gamma conjugacy"));
        }
    }
    for (prior, name) in config
        .variance_priors
        .iter()
        .zip(&config.variance_names)
        .chain(std::iter::once((
            &config.observation_variance_prior,
            &"observation_variance".to_string(),
        )))
    {
        prior.validate(&format!("{name} prior"))?;
    }
    let augmented = d.saturating_add(config.coefficient_prior.mean.len());
    schedule.check_fit_size(
        "regression fit",
        &[augmented.saturating_add(config.variance_priors.len() + 1)],
        &[y.len() + 1, augmented, augmented, 3],
    )?;
    let template = config.structural_model.with_static_regression(
        design,
        &config.coefficient_prior.mean,
        &config.coefficient_prior.covariance,
    )?;
    let chains = run_gibbs_chains(
        &schedule,
        config.seed,
        FIT_SEED_DOMAIN,
        |rng| {
            let variances: Vec<f64> = config
                .variance_priors
                .iter()
                .map(|prior| overdispersed_positive(prior.mode(), rng))
                .collect();
            let observation_variance =
                overdispersed_positive(config.observation_variance_prior.mode(), rng);
            Ok::<_, StateSpaceError>((template.clone(), variances, observation_variance))
        },
        |(model, variances, observation_variance), rng, retain| {
            model.set_variances(&config.innovation_indices, variances, *observation_variance);
            let states = model.sample_states_ffbs(y, rng)?;
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
                *variance = draw_variance(prior, y.len(), ss, rng)?;
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
            *observation_variance =
                draw_variance(config.observation_variance_prior, observed, ss, rng)?;
            Ok(retain.then(|| {
                let state = states.last().expect("validated nonempty data");
                RegressionDraw {
                    variances: variances.clone(),
                    observation_variance: *observation_variance,
                    coefficients: state[d..].to_vec(),
                    terminal_state: state[..d].to_vec(),
                }
            }))
        },
    )?;
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
        check_forecast_size("regression forecast", &self.chains, design.len(), 6)?;
        let per_draw = simulate_draws(
            &self.chains,
            seed,
            FORECAST_SEED_DOMAIN,
            |_, _, draw: &RegressionDraw, rng| {
                let mut simulation = self.config.structural_model.clone();
                simulation.set_variances(
                    &self.config.innovation_indices,
                    &draw.variances,
                    draw.observation_variance,
                );
                let mut state = draw.terminal_state.clone();
                let (mut l, mut s, mut r, mut m, mut o, mut c) =
                    (vec![], vec![], vec![], vec![], vec![], vec![]);
                let mut total = 0.0;
                for row in design {
                    let next = simulation.simulate_transition(&state, rng)?;
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
                    let observation = mean + normal(rng) * draw.observation_variance.sqrt();
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
                Ok([l, s, r, m, o, c])
            },
        )?;
        let [level_paths, secondary_paths, regression_paths, mean_paths, observation_paths, cumulative_observation_paths] =
            split_paths(per_draw);
        Ok(RegressionForecast {
            level_paths,
            secondary_paths,
            regression_paths,
            mean_paths,
            observation_paths,
            cumulative_observation_paths,
        })
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
    use crate::seeding::chain_seed;
    use rand::SeedableRng;

    #[test]
    fn fit_and_forecast_seed_domains_are_distinct() {
        for chain in 0..4 {
            assert_ne!(
                chain_seed(42, chain, FIT_SEED_DOMAIN),
                chain_seed(42, chain, FORECAST_SEED_DOMAIN)
            );
        }
    }
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
    /// Exact `P(V <= v)` for `V ~ InverseGamma(3, scale)`. At integer shape the
    /// upper incomplete gamma closes in elementary terms, so the prior mass
    /// inside an acceptance window can be stated rather than guessed at.
    fn inverse_gamma3_cdf(scale: f64, v: f64) -> f64 {
        let t = scale / v;
        (-t).exp() * (1.0 + t + t * t / 2.0)
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
            // Mean 0.6, six times the observation variance the recovery test
            // simulates, and 1.4% of its mass inside that test's acceptance
            // window: the fit cannot meet the window by echoing this prior back.
            // The cost of moving it there is small and one-directional - the
            // conjugate update adds `scale / (shape + n / 2 - 1)`, so raising
            // the scale from 0.4 to 1.2 pushes the posterior mean up by about
            // 0.006 at that test's 250 rows, an eighth of its window. See
            // `seeded_recovery_and_pool_independence`.
            observation_variance_prior: InverseGammaPrior {
                shape: 3.0,
                scale: 1.2,
            },
            coefficient_prior: GaussianCoefficientPrior {
                mean: vec![0.0],
                covariance: vec![9.0],
            },
            num_chains: 2,
            num_draws: 500,
            num_warmup: 250,
            thinning: 1,
            seed: 32,
            seasonal: false,
        }
    }
    #[test]
    fn seeded_recovery_and_pool_independence() {
        let mut rng = ChaCha8Rng::seed_from_u64(188);
        let mut level = 0.0;
        let x: Vec<Vec<f64>> = (0..250).map(|_| vec![normal(&mut rng)]).collect();
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
        // The coefficient prior is N(0, 9), so a fit that ignored the series
        // would report 0. Asserted at run time rather than stated in a comment,
        // so widening the window back onto the prior turns this red.
        assert!(
            (0.0f64 - 1.8).abs() > 2.0 * 0.12,
            "the coefficient prior mean is within one tolerance-width of the window"
        );
        assert!((mean - 1.8).abs() < 0.12, "{mean}");
        let obs = fit
            .chains
            .iter()
            .flatten()
            .map(|d| d.observation_variance)
            .sum::<f64>()
            / coefficients.len() as f64;
        // The window this replaced, `0.05..0.2`, held 66.3% of the
        // InverseGamma(3, 0.4) prior it was checked against, and that prior's
        // median of 0.1496 sat inside it. The assertion is on an average of
        // hundreds of draws rather than on one, so the relevant figure is where
        // the prior mean falls: at 0.2, exactly the window's excluded upper
        // edge, which makes a prior-only fit a coin flip rather than a certain
        // failure. Both the prior mean and the prior mass inside the window are
        // now asserted to stay clear of it, which keeps a future widening
        // honest.
        const OBSERVATION_VARIANCE: f64 = 0.1;
        const WINDOW: f64 = 0.05;
        let prior = config().observation_variance_prior;
        assert_eq!(
            prior.shape, 3.0,
            "the closed-form prior CDF below assumes shape 3"
        );
        let prior_mean = prior.scale / (prior.shape - 1.0);
        assert!(
            (prior_mean - OBSERVATION_VARIANCE).abs() > 2.0 * WINDOW,
            "prior mean {prior_mean} is not clear of the acceptance window"
        );
        let prior_mass = inverse_gamma3_cdf(prior.scale, OBSERVATION_VARIANCE + WINDOW)
            - inverse_gamma3_cdf(prior.scale, OBSERVATION_VARIANCE - WINDOW);
        assert!(
            prior_mass < 0.02,
            "the observation-variance prior puts {prior_mass} of its mass inside the \
             acceptance window, so the window does not demonstrate recovery"
        );
        assert!((obs - OBSERVATION_VARIANCE).abs() < WINDOW, "{obs}");
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
    fn forecast_keeps_fixed_correlated_process_noise() {
        let mut c = config();
        c.structural_model = LinearGaussianStateSpace::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.8, 0.3, 0.3, 0.6],
            1.0,
            vec![0.0; 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        c.innovation_indices.clear();
        c.variance_priors.clear();
        c.variance_names.clear();
        let draw = RegressionDraw {
            variances: vec![],
            observation_variance: 1.0,
            coefficients: vec![0.0],
            terminal_state: vec![0.0; 2],
        };
        let post = RegressionPosterior {
            config: c,
            chains: vec![vec![draw; 15000]],
        };
        let paths = post.forecast(&[vec![0.0], vec![0.0]], 81).unwrap();
        let product = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>, i: usize, j: usize| {
            a.iter().zip(b).map(|(x, y)| x[i] * y[j]).sum::<f64>() / a.len() as f64
        };
        assert!((product(&paths.level_paths[0], &paths.level_paths[0], 0, 0) - 0.8).abs() < 0.035);
        assert!(
            (product(&paths.level_paths[0], &paths.secondary_paths[0], 0, 0) - 0.3).abs() < 0.035
        );
        assert!((product(&paths.level_paths[0], &paths.level_paths[0], 0, 1) - 0.8).abs() < 0.045);
    }
    #[test]
    fn independent_inverse_gamma_updates_reject_correlations_and_duplicate_indices() {
        let mut c = config();
        c.innovation_indices = vec![0, 0];
        c.variance_priors.push(c.variance_priors[0]);
        c.variance_names.push("duplicate".into());
        assert!(fit_regression(&[1.0, 2.0], &[vec![1.0], vec![1.0]], &c).is_err());
        let mut c = config();
        c.structural_model = LinearGaussianStateSpace::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 0.3, 0.3, 1.0],
            1.0,
            vec![0.0; 2],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert!(fit_regression(&[1.0, 2.0], &[vec![1.0], vec![1.0]], &c).is_err());
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
