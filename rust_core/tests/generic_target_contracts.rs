use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::Graph;

#[test]
fn large_matrix_gradient_is_identical_across_worker_counts() {
    let mut graph = Graph::new();
    let cols = 25;
    let rows = 4000;
    let start = graph.add_vector_params("beta", cols);
    graph.vector_normal_logp(start, cols, 0.0, 1.0);
    let matrix: Vec<_> = (0..rows * cols)
        .map(|i| ((i * 17 % 101) as f64 - 50.0) / 37.0)
        .collect();
    let slot = graph.store_named_matrix("X", matrix, rows, cols);
    let mean = graph.mat_vec_mul(slot, start, cols, None);
    let obs = graph.add_named_obs_data(
        "y",
        "obs",
        (0..rows).map(|i| (i % 13) as f64 / 7.0).collect(),
    );
    let sigma = graph.add_constant(1.3);
    graph.normal_obs_logp(mean, sigma, obs);
    let params: Vec<_> = (0..cols).map(|i| (i as f64 - 12.0) / 41.0).collect();
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut eval = Evaluator::new(&graph);
                eval.compute(&graph, &params);
                (eval.total_logp, eval.grad)
            })
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn batch_initial_positions_follow_ids_and_collect_invalid_cells() {
    use rustmc_core::data::DataBinding;
    use rustmc_core::graph::ElementwiseOp;
    use rustmc_core::sampler::{
        sample_batch_bound_with_initial, BatchSampleConfig, BoundBatchOptions,
    };
    use std::collections::HashMap;
    use std::sync::Arc;
    let mut graph = Graph::new();
    let parameter = rustmc_core::distributions::Normal::prior(&mut graph, "x", 0.0, 1.0);
    let positive = graph.elementwise(ElementwiseOp::Log, parameter, None);
    graph.add_logp_term(positive);
    let binding = DataBinding::from_graph(&graph).unwrap();
    let graph = Arc::new(graph.structure_only());
    let config = BatchSampleConfig {
        num_chains: 2,
        num_draws: 5,
        num_warmup: 5,
        ..Default::default()
    };
    let options = BoundBatchOptions {
        threads: 2,
        collect_errors: true,
        ..Default::default()
    };
    let result = sample_batch_bound_with_initial(
        Arc::clone(&graph),
        vec![
            ("a".into(), Ok(binding.clone())),
            ("b".into(), Ok(binding.clone())),
        ],
        config.clone(),
        options.clone(),
        HashMap::from([("a".into(), vec![vec![1.0], vec![2.0]])]),
    )
    .unwrap();
    assert!(result[0].is_ok());
    assert!(result[1].as_ref().unwrap_err().contains("initial"));
    assert!(sample_batch_bound_with_initial(
        graph,
        vec![("a".into(), Ok(binding))],
        config,
        options,
        HashMap::from([("typo".into(), vec![vec![1.0], vec![2.0]])])
    )
    .unwrap_err()
    .contains("unknown dataset ID"));
}

#[test]
fn immutable_binding_sharing_compares_allocations_not_values() {
    use rustmc_core::data::{DataBinding, DataInputs, MatrixBinding};
    use std::collections::HashMap;
    use std::sync::Arc;
    let mut graph = Graph::new();
    graph.store_named_matrix("X", vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let values: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0]);
    let inputs = DataInputs {
        vectors: HashMap::new(),
        matrices: HashMap::from([(
            "X".into(),
            MatrixBinding {
                data: values,
                n_rows: 2,
                n_cols: 2,
            },
        )]),
    };
    let first = DataBinding::bind(&graph.schema, inputs.clone(), "a", true, true).unwrap();
    let second = DataBinding::bind(&graph.schema, inputs.clone(), "b", true, true).unwrap();
    assert!(first.shares_payload_with(&second, "X"));
    let mut copied = inputs;
    copied.matrices.get_mut("X").unwrap().data = Arc::from(vec![1.0, 2.0, 3.0, 4.0]);
    let copied = DataBinding::bind(&graph.schema, copied, "c", true, true).unwrap();
    assert!(!first.shares_payload_with(&copied, "X"));
    assert!(!first.shares_payload_with(&second, "missing"));
}
