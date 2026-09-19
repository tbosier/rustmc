use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::DataBinding;
use rustmc_core::distributions::{LogNormal, Uniform};
use rustmc_core::graph::{Graph, ParamTransform};
use rustmc_core::sampler::{sample_bound_with_init, SamplerConfig};
use std::sync::Arc;

#[test]
fn scalar_and_vector_uniform_reject_invalid_ranges() {
    for (lower, upper) in [
        (1.0, 1.0),
        (2.0, 1.0),
        (-1e308, 1e308),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
    ] {
        for scalar in [false, true] {
            let mut graph = Graph::new();
            if scalar {
                Uniform::prior(&mut graph, "x", lower, upper);
            } else {
                let start = graph.add_vector_params_with_transform(
                    "x",
                    1,
                    ParamTransform::BoundedSigmoid { lower, upper },
                );
                graph.vector_uniform_logp(start, 1, lower, upper);
            }
            for raw in [-40.0, 0.0, 40.0] {
                let mut evaluator = Evaluator::new(&graph);
                evaluator.compute(&graph, &[raw]);
                assert_eq!(evaluator.total_logp, f64::NEG_INFINITY);
                assert_eq!(evaluator.grad, vec![0.0]);
            }
        }
    }
}

#[test]
fn native_sampler_rejects_overflowing_constrained_draws() {
    let mut graph = Graph::new();
    LogNormal::prior(&mut graph, "x", 800.0, 1.0);
    let binding = DataBinding::from_graph(&graph).unwrap();
    let result = sample_bound_with_init(
        Arc::new(graph),
        binding,
        SamplerConfig {
            num_chains: 1,
            num_draws: 2,
            num_warmup: 1,
            step_size: 1e-6,
            show_progress: false,
            ..Default::default()
        },
        Some(vec![vec![800.0]]),
    );
    assert!(result
        .unwrap_err()
        .contains("nonfinite after transformation"));
}
