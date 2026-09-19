use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::DataBinding;
use rustmc_core::distributions::{LogNormal, Normal, Uniform};
use rustmc_core::graph::{Graph, ParamTransform};
use rustmc_core::sampler::{sample, sample_bound_with_init, SamplerConfig};
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

/// A graph that names a data slot it has no payload for is rejected, not a
/// panic.
///
/// `broadcast_observation` and `fused_linear_mu` carry raw indices into the
/// binding's observation and vector lists. Nothing created an `Op::Data` node
/// for those slots, so the check that a binding covers everything the graph
/// reads did not count them, and `validate_node_lengths` indexed an empty list.
/// `sampler::sample` validates shapes before it does anything else, so the
/// panic came out of `sample` in place of the `Result` it promises. Both
/// builders are public and this graph is built entirely through them.
#[test]
fn a_data_slot_the_graph_names_but_does_not_own_is_an_error_not_a_panic() {
    let dangling_observation = || {
        let mut graph = Graph::new();
        let p = Normal::prior(&mut graph, "p", 0.0, 1.0);
        graph.broadcast_observation(p, 0);
        graph
    };
    let message = dangling_observation()
        .validate_shapes()
        .expect_err("a graph that names an observation slot it has no payload for is invalid")
        .to_string();
    assert!(
        message.contains("binding does not provide every data slot"),
        "got {message}"
    );

    let error = sample(dangling_observation(), SamplerConfig::default())
        .expect_err("and the sampler must return that, not unwind");
    assert!(
        error.contains("binding does not provide every data slot"),
        "got {error}"
    );

    // The same hole on the vector side: `store_data_vec` registers no schema
    // slot, so a `FusedLinearMu` past the end of the binding was uncounted too.
    let mut graph = Graph::new();
    let p = Normal::prior(&mut graph, "p", 0.0, 1.0);
    let column = graph.store_data_vec(vec![0.1, 0.2, 0.3]);
    graph.fused_linear_mu(vec![p], vec![column + 4], None);
    let message = graph
        .validate_shapes()
        .expect_err("a predictor column past the end of the binding is invalid")
        .to_string();
    assert!(
        message.contains("binding does not provide every data slot"),
        "got {message}"
    );
}
