use rustmc_core::autodiff::Evaluator;
use rustmc_core::distributions::Normal;
use rustmc_core::graph::{Graph, ObsFamily};
use rustmc_core::observation;
use rustmc_core::sampler::{sample, SamplerConfig};

fn graph(count: f64, standardized: bool) -> Graph {
    let mut graph = Graph::new();
    let eta = if standardized {
        let z = Normal::prior(&mut graph, "z", 0.0, 10.0);
        let mean = graph.add_constant(count.ln());
        let scale = graph.add_constant(1.0 / count.sqrt());
        let shift = graph.mul(z, scale);
        graph.add(mean, shift)
    } else {
        graph.add_param("eta")
    };
    let observed = graph.add_obs_data(vec![count]);
    let means = graph.broadcast_observation(eta, observed);
    graph.obs_logp_poisson_log(means, observed);
    graph
}

#[test]
fn high_rate_poisson_density_curvature_and_gradients_are_preserved() {
    for count in [1e14_f64, 1e15, 8e15] {
        let graph = graph(count, false);
        let mut evaluator = Evaluator::new(&graph);
        let expected_mode = -0.5 * (std::f64::consts::TAU.ln() + count.ln()) - 1.0 / (12.0 * count);
        let center = count.ln();
        let sd = 1.0 / count.sqrt();
        for z in [-1.0, 0.0, 1.0] {
            let eta = center + z * sd;
            evaluator.compute(&graph, &[eta]);
            assert!((evaluator.total_logp - (expected_mode - 0.5 * z * z)).abs() < 3e-6);
            assert_eq!(evaluator.grad[0], count - eta.exp());
            let pointwise =
                observation::log_density(ObsFamily::PoissonLog, count, eta, None).unwrap();
            assert_eq!(pointwise, evaluator.total_logp);
            if z != 0.0 {
                let h = 0.05 * sd;
                evaluator.compute(&graph, &[eta + h]);
                let plus = evaluator.total_logp;
                evaluator.compute(&graph, &[eta - h]);
                let minus = evaluator.total_logp;
                assert!(((plus - minus) / (2.0 * h) / (count - eta.exp()) - 1.0).abs() < 2e-5);
            }
        }
    }
}

#[test]
fn poisson_log_rate_tails_keep_finite_log_densities_when_rates_underflow() {
    for count in [0.0, 1.0, 20.0] {
        for eta in [-740.0, -1000.0] {
            let graph = graph(count, false);
            let mut evaluator = Evaluator::new(&graph);
            evaluator.compute(&graph, &[eta]);
            let expected = count * eta - rustmc_core::autodiff::ln_gamma(count + 1.0);
            assert!((evaluator.total_logp - expected).abs() < 1e-10);
            assert!(evaluator.total_logp.is_finite());
        }
    }
    assert_eq!(
        observation::log_density(ObsFamily::PoissonLog, 1.0, 1000.0, None).unwrap(),
        f64::NEG_INFINITY
    );
}

#[test]
fn high_rate_poisson_posterior_recovers_local_gaussian_width() {
    // Four chains rather than one: across 30 seeds a single 3000-draw chain
    // put the variance outside this tolerance once (0.861 at seed 913), which
    // is sampling noise, not a density error. Pooling chains shrinks that noise
    // without widening the tolerance.
    let result = sample(
        graph(8e15, true),
        SamplerConfig {
            num_chains: 4,
            num_draws: 3000,
            num_warmup: 500,
            seed: 913,
            show_progress: false,
            ..Default::default()
        },
    )
    .unwrap();
    let samples = result
        .samples
        .iter()
        .flatten()
        .map(|draw| draw[0])
        .collect::<Vec<_>>();
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    assert!(mean.abs() < 0.12, "mean {mean}");
    assert!((variance - 1.0 / 1.01).abs() < 0.12, "variance {variance}");
}
