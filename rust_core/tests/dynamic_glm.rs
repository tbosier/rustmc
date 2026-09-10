use rustmc_core::dynamic_glm::*;

fn cfg(family: Family) -> DynamicGlmConfig {
    DynamicGlmConfig {
        family,
        chains: 2,
        draws: 2500,
        warmup: 500,
        process_sd: 0.,
        group_sd: 0.,
        ..Default::default()
    }
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

#[test]
fn poisson_intercept_matches_independent_grid_posterior_with_exposure() {
    let config = cfg(Family::Poisson);
    let y = vec![vec![0., 1., 4., f64::NAN, 2.]];
    let exposure = vec![vec![0., 0.5, 2., 100., 1.]];
    let fit = fit_dynamic_glm(&y, None, Some(&exposure), &config).unwrap();
    let values: Vec<_> = fit
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][0][0])
        .collect();
    let mut norm = 0.;
    let mut first = 0.;
    let mut second = 0.;
    for i in 0..20001 {
        let b = -8. + i as f64 * 0.0008;
        let weight = (-0.5 * b * b + 7. * b - 3.5 * b.exp()).exp();
        norm += weight;
        first += weight * b;
        second += weight * b * b;
    }
    let reference = first / norm;
    assert!((mean(&values) - reference).abs() < 0.04);
    assert!((variance(&values) - (second / norm - reference * reference)).abs() < 0.025);
    assert_eq!(fit.observed_count, 4);
    assert!(fit.likelihood_evaluations.iter().all(|n| *n > config.draws));
    let forecast = fit
        .forecast(2, None, Some(&vec![vec![0., 1.]]), 88)
        .unwrap();
    assert!(forecast
        .observation_paths
        .iter()
        .flatten()
        .all(|d| d[0][0] == 0.));
}

#[test]
fn gaussian_sparse_group_shrinks_and_joint_draws_match_analytic_covariance() {
    let mut config = cfg(Family::Gaussian);
    config.group_sd = 0.4;
    config.draws = 7000;
    let y = vec![vec![2., 2., 2., 2.], vec![f64::NAN; 4]];
    let fit = fit_dynamic_glm(&y, None, None, &config).unwrap();
    let dense: Vec<_> = fit
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][0][0])
        .collect();
    let sparse: Vec<_> = fit
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][1][0])
        .collect();
    // Prior Var(beta_g)=1.16, Cov(beta_0,beta_1)=1; observed mean variance .25.
    let expected_dense = 2. * 1.16 / 1.41;
    let expected_sparse = 2. / 1.41;
    assert!((mean(&dense) - expected_dense).abs() < 0.06);
    assert!((mean(&sparse) - expected_sparse).abs() < 0.06);
    let covariance = dense
        .iter()
        .zip(&sparse)
        .map(|(a, b)| (a - mean(&dense)) * (b - mean(&sparse)))
        .sum::<f64>()
        / dense.len() as f64;
    assert!(
        (covariance - (1. - 1.16 / 1.41)).abs() < 0.05,
        "{covariance}"
    );
    let f = fit.forecast(1, None, None, 8).unwrap();
    let a: Vec<_> = f.mean_paths.iter().flatten().map(|d| d[0][0]).collect();
    let b: Vec<_> = f.mean_paths.iter().flatten().map(|d| d[1][0]).collect();
    let total: Vec<_> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
    assert!(variance(&total) > variance(&a) + variance(&b) + 0.2);
}

fn known_posterior(family: Family, draws: usize) -> DynamicGlmPosterior {
    let mut config = cfg(family);
    config.chains = 1;
    config.draws = draws;
    config.observation_sd = 0.6;
    config.dispersion = 2.;
    let components = if family == Family::HurdleLogNormal {
        2
    } else {
        1
    };
    let population_coefficients = (0..components)
        .map(|c| {
            vec![if c == 0 {
                2f64.ln()
            } else {
                (0.3f64 / 0.7).ln()
            }]
        })
        .collect::<Vec<_>>();
    let draw = DynamicGlmDraw {
        coefficients: population_coefficients
            .iter()
            .map(|b| vec![b.clone(), b.clone()])
            .collect(),
        population_coefficients,
        states: vec![vec![vec![0.], vec![0.]]; components],
    };
    DynamicGlmPosterior {
        config,
        chains: vec![vec![draw; draws]],
        groups: 2,
        time_count: 1,
        features: 0,
        likelihood_evaluations: vec![0],
        observed_count: 0,
    }
}

#[test]
fn nb_predictive_mean_overdispersion_and_tail_match_reference() {
    let p = known_posterior(Family::NegativeBinomial, 60000);
    let f = p.forecast(1, None, None, 401).unwrap();
    let samples: Vec<_> = f.observation_paths[0].iter().map(|d| d[0][0]).collect();
    assert!((mean(&samples) - 2.).abs() < 0.04);
    assert!((variance(&samples) - 4.).abs() < 0.15);
    // NB(r=2,p=.5): P(Y>=6)=sum_{k>=6}(k+1)/2^(k+2)=.0625.
    let tail = samples.iter().filter(|x| **x >= 6.).count() as f64 / samples.len() as f64;
    assert!((tail - 0.0625).abs() < 0.004);
}

#[test]
fn hurdle_forecast_zero_frequency_mean_and_tail_match_known_model() {
    let p = known_posterior(Family::HurdleLogNormal, 60000);
    let f = p.forecast(1, None, None, 70).unwrap();
    let samples: Vec<_> = f.observation_paths[0].iter().map(|d| d[0][0]).collect();
    let zeros = samples.iter().filter(|x| **x == 0.).count() as f64 / samples.len() as f64;
    assert!((zeros - 0.7).abs() < 0.006);
    assert!((mean(&samples) - 0.3 * 2. * 0.18f64.exp()).abs() < 0.025);
    let tail =
        samples.iter().filter(|x| **x > 2. * 1.2f64.exp()).count() as f64 / samples.len() as f64;
    assert!((tail - 0.3 * 0.0227501319).abs() < 0.0015);
    assert!((f.positive_mean_paths[0][0][0][0] - 2. * 0.18f64.exp()).abs() < 1e-12);
}

#[test]
fn shared_dynamic_shocks_produce_correct_horizon_covariance() {
    let mut p = known_posterior(Family::Gaussian, 40000);
    p.config.shared_process_sd = 0.4;
    p.config.process_sd = 0.2;
    let f = p.forecast(3, None, None, 480).unwrap();
    let a: Vec<_> = f.mean_paths[0].iter().map(|d| d[0][2]).collect();
    let b: Vec<_> = f.mean_paths[0].iter().map(|d| d[1][2]).collect();
    let ma = mean(&a);
    let mb = mean(&b);
    let cov = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - ma) * (b - mb))
        .sum::<f64>()
        / a.len() as f64;
    assert!((cov - 3. * 0.16).abs() < 0.02);
    assert!((variance(&a) - 3. * 0.20).abs() < 0.025);
}

#[test]
fn missing_zero_semantics_and_artifact_validation() {
    let mut config = cfg(Family::HurdleLogNormal);
    config.draws = 300;
    config.warmup = 100;
    config.process_sd = 0.1;
    let y = vec![vec![0., f64::NAN, 0., 0.]];
    let p = fit_dynamic_glm(&y, None, None, &config).unwrap();
    assert_eq!(p.observed_count, 3);
    let f = p.forecast(2, None, None, 7).unwrap();
    assert!(f
        .occurrence_paths
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .all(|x| (0. ..=1.).contains(x)));
    let json = p.to_json(&y).unwrap();
    let (restored, observations) = DynamicGlmPosterior::from_json(&json).unwrap();
    assert!(observations[0][1].is_nan());
    assert_eq!(f, restored.forecast(2, None, None, 7).unwrap());
    assert!(
        DynamicGlmPosterior::from_json(&json.replace("\"version\":1", "\"version\":99")).is_err()
    );
    let mut damaged: serde_json::Value = serde_json::from_str(&json).unwrap();
    damaged["posterior"]["groups"] = serde_json::json!(2);
    assert!(DynamicGlmPosterior::from_json(&damaged.to_string()).is_err());
    config.family = Family::Poisson;
    assert!(fit_dynamic_glm(&vec![vec![1.]], None, Some(&vec![vec![0.]]), &config).is_err());
    assert!(fit_dynamic_glm(&vec![vec![0.5]], None, None, &config).is_err());
}

#[test]
fn count_regression_and_dynamic_level_recover_simulated_signal() {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Poisson};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12);
    let x = vec![(0..36)
        .map(|t| vec![if t % 2 == 0 { -1. } else { 1. }])
        .collect::<Vec<_>>()];
    let y = vec![(0..36)
        .map(|t| {
            Poisson::new((1.2 + 0.65 * x[0][t][0] + 0.025 * t as f64).exp())
                .unwrap()
                .sample(&mut rng)
        })
        .collect()];
    let mut config = cfg(Family::Poisson);
    config.process_sd = 0.08;
    config.draws = 1500;
    config.warmup = 700;
    let p = fit_dynamic_glm(&y, Some(&x), None, &config).unwrap();
    let beta: Vec<_> = p
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][0][1])
        .collect();
    assert!((mean(&beta) - 0.65).abs() < 0.2, "{}", mean(&beta));
    let change: Vec<_> = p
        .chains
        .iter()
        .flatten()
        .map(|d| d.states[0][0][35] - d.states[0][0][0])
        .collect();
    assert!(mean(&change) > 0.2, "{}", mean(&change));
    assert!(p.forecast(1, None, None, 2).is_err());
}

#[test]
fn hurdle_covariates_recover_distinct_occurrence_and_positive_effects() {
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(314);
    let x = vec![(0..80)
        .map(|t| vec![if t % 2 == 0 { -1. } else { 1. }])
        .collect::<Vec<_>>()];
    let y = vec![(0..80)
        .map(|t| {
            let predictor = x[0][t][0];
            let probability =
                1. / (1. + (-(-0.2 + 1.2 * predictor + 0.02 * (t as f64 - 40.))).exp());
            if rng.gen::<f64>() < probability {
                let noise: f64 = StandardNormal.sample(&mut rng);
                (0.8 + 0.7 * predictor + 0.4 * noise).exp()
            } else {
                0.
            }
        })
        .collect()];
    let mut config = cfg(Family::HurdleLogNormal);
    config.process_sd = 0.05;
    config.observation_sd = 0.4;
    config.draws = 1800;
    config.warmup = 800;
    let p = fit_dynamic_glm(&y, Some(&x), None, &config).unwrap();
    let severity: Vec<_> = p
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][0][1])
        .collect();
    let occurrence: Vec<_> = p
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[1][0][1])
        .collect();
    assert!(
        (mean(&severity) - 0.7).abs() < 0.2,
        "severity {}",
        mean(&severity)
    );
    assert!(mean(&occurrence) > 0.6, "occurrence {}", mean(&occurrence));
    let future = vec![vec![vec![-1.], vec![1.]]];
    let f = p.forecast(2, Some(&future), None, 40).unwrap();
    assert!(
        f.occurrence_paths
            .iter()
            .flatten()
            .map(|d| d[0][1] - d[0][0])
            .sum::<f64>()
            > 0.
    );
    assert!(
        f.positive_mean_paths
            .iter()
            .flatten()
            .map(|d| d[0][1] - d[0][0])
            .sum::<f64>()
            > 0.
    );
}

#[test]
fn nb_intercept_matches_grid_and_chains_are_thread_invariant() {
    let config = cfg(Family::NegativeBinomial);
    let y = vec![vec![0., 1., 4., 2.]];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| fit_dynamic_glm(&y, None, None, &config).unwrap())
    };
    let p = run(1);
    assert_eq!(p.chains, run(3).chains);
    let samples: Vec<_> = p
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][0][0])
        .collect();
    let mut norm = 0.;
    let mut first = 0.;
    for i in 0..20001 {
        let b = -8. + i as f64 * 0.0008;
        // Independent NB mean parametrization, dropping constants in b.
        let likelihood = 7. * b - (4. * 5. + 7.) * (5. + b.exp()).ln();
        let weight = (-0.5 * b * b + likelihood + 50.).exp();
        norm += weight;
        first += weight * b;
    }
    assert!((mean(&samples) - first / norm).abs() < 0.04);
}

#[test]
fn all_zero_severity_retains_gaussian_prior_and_prior_predictive_timing() {
    let mut config = cfg(Family::HurdleLogNormal);
    config.draws = 5000;
    config.process_sd = 0.1;
    let p = fit_dynamic_glm(&vec![vec![0.; 5]], None, None, &config).unwrap();
    let severity: Vec<_> = p
        .chains
        .iter()
        .flatten()
        .map(|d| d.coefficients[0][0][0] + d.states[0][0][4])
        .collect();
    assert!(mean(&severity).abs() < 0.06);
    assert!((variance(&severity) - 1.05).abs() < 0.08);
    config.family = Family::Gaussian;
    config.draws = 40000;
    let f = prior_predictive(&config, 1, 3, None, None).unwrap();
    let first: Vec<_> = f.mean_paths.iter().flatten().map(|d| d[0][0]).collect();
    let third: Vec<_> = f.mean_paths.iter().flatten().map(|d| d[0][2]).collect();
    assert!((variance(&first) - 1.01).abs() < 0.03);
    assert!((variance(&third) - 1.03).abs() < 0.03);
}

#[test]
fn repeated_prior_simulations_have_reasonable_count_posterior_coverage() {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma, Poisson, StandardNormal};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(8517);
    let mut covered = 0;
    let mut ranks = Vec::new();
    for replicate in 0..16 {
        let family = if replicate % 2 == 0 {
            Family::Poisson
        } else {
            Family::NegativeBinomial
        };
        let truth: f64 = StandardNormal.sample(&mut rng);
        let y = vec![(0..12)
            .map(|_| {
                let rate = if family == Family::Poisson {
                    truth.exp()
                } else {
                    Gamma::new(5., truth.exp() / 5.).unwrap().sample(&mut rng)
                };
                Poisson::new(rate).unwrap().sample(&mut rng)
            })
            .collect()];
        let mut config = cfg(family);
        config.draws = 1000;
        config.warmup = 500;
        config.seed = 100 + replicate;
        let p = fit_dynamic_glm(&y, None, None, &config).unwrap();
        let values: Vec<_> = p
            .chains
            .iter()
            .flatten()
            .map(|d| d.coefficients[0][0][0])
            .collect();
        let rank = values.iter().filter(|b| **b < truth).count() as f64 / values.len() as f64;
        if (0.05..=0.95).contains(&rank) {
            covered += 1;
        }
        ranks.push(rank);
    }
    // Small repeated-simulation smoke gate; not a claim of uniform SBC ranks.
    assert!(
        covered >= 12,
        "90% intervals covered {covered}/16; ranks={ranks:?}"
    );
    assert!((0.25..0.75).contains(&mean(&ranks)), "ranks={ranks:?}");
}

#[test]
fn artifact_observations_must_obey_family_support() {
    for family in [
        Family::Poisson,
        Family::NegativeBinomial,
        Family::HurdleLogNormal,
    ] {
        let mut config = cfg(family);
        config.chains = 1;
        config.draws = 2;
        config.warmup = 1;
        let y = vec![vec![0.]];
        let p = fit_dynamic_glm(&y, None, None, &config).unwrap();
        let invalid_values = if family == Family::HurdleLogNormal {
            vec![-1.0]
        } else {
            vec![-1.0, 0.5, 9_007_199_254_740_992.0]
        };
        for value in invalid_values {
            assert!(p.to_json(&vec![vec![value]]).is_err());
            let mut archive: serde_json::Value =
                serde_json::from_str(&p.to_json(&y).unwrap()).unwrap();
            archive["observations"][0][0] = serde_json::json!(value);
            assert!(DynamicGlmPosterior::from_json(&archive.to_string()).is_err());
        }
    }
}
#[test]
fn posterior_must_preserve_zero_scale_constraints() {
    let p = known_posterior(Family::Gaussian, 2);
    let mut invalid = p.clone();
    invalid.chains[0][0].coefficients[0][0][0] += 1.;
    assert!(invalid.forecast(1, None, None, 42).is_err());
    let mut invalid = p.clone();
    invalid.chains[0][0].states[0][0][0] = 1.;
    assert!(invalid.forecast(1, None, None, 42).is_err());
    let mut invalid = p.clone();
    invalid.config.shared_process_sd = 0.1;
    invalid.chains[0][0].states[0][0][0] = 1.;
    assert!(invalid.forecast(1, None, None, 42).is_err());
}
#[test]
fn extreme_prior_dimensions_return_errors_without_panicking() {
    let mut config = cfg(Family::Gaussian);
    config.chains = 1;
    config.draws = 1;
    assert!(prior_predictive(&config, usize::MAX, 1, None, None).is_err());
    assert!(prior_predictive(&config, 1, usize::MAX, None, None).is_err());
}
