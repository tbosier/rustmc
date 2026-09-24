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

const GROUPED_MODEL: &str = r#"{"format":"rustmc.graph-model","version":1,"definition":{"dimensions":{},"potentials":[],"deterministics":[],"priors":[{"Normal":{"name":"mu","mu":{"Const":0.0},"sigma":{"Const":1.0}}},{"VectorNormal":{"name":"z","n":3,"mu":0.0,"sigma":1.0}}],"likelihoods":[{"family":"Normal","name":"obs","mu_expr":{"Add":[{"Param":"mu"},{"Gather":{"param_name":"z","data_key":"site"}}]},"sigma":{"Const":1.0},"observed_key":"y"}]},"schema":{"vectors":[{"key":"site","kind":"Vector","dim":"obs"}],"observations":[{"key":"y","kind":{"Observation":{"likelihood":"obs"}},"dim":"obs"}],"matrices":[]}}"#;

fn grouped_inputs(site: Vec<f64>, observed: bool) -> rustmc_core::data::DataInputs {
    let mut inputs = rustmc_core::data::DataInputs::default();
    if observed {
        inputs
            .vectors
            .insert("y".into(), Arc::from(vec![0.5; site.len()]));
    }
    inputs.vectors.insert("site".into(), Arc::from(site));
    inputs
}

/// Group indices outside the group count used to pass binding validation and
/// reach `Evaluator::new`'s `expect` from `GraphModel::log_density`,
/// `ModelFit::predict` and prior-predictive simulation.
#[test]
fn out_of_range_group_indices_are_errors_not_panics() {
    use rustmc_core::model::GraphModel;
    use std::collections::HashMap;
    let model = GraphModel::from_json(GROUPED_MODEL).unwrap();
    for site in [vec![0.0, 3.0], vec![-1.0, 0.0], vec![0.5, 1.0]] {
        let error = model
            .bind(grouped_inputs(site.clone(), true), "bad")
            .expect_err("indices outside [0, 3) must not bind");
        assert!(error.to_string().contains("group indices"), "{error}");
    }
    let good = model
        .bind(grouped_inputs(vec![0.0, 1.0, 2.0], true), "fit")
        .unwrap();
    let fit = model
        .sample(
            good,
            SamplerConfig {
                num_chains: 1,
                num_draws: 5,
                num_warmup: 20,
                show_progress: false,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let error = fit
        .predict(
            grouped_inputs(vec![0.0, 7.0], false),
            HashMap::new(),
            1,
            false,
        )
        .expect_err("prediction at an unknown group must be refused");
    assert!(error.to_string().contains("group"), "{error}");
}

/// An empty data vector in a hand-built graph used to classify the vector op
/// reading it as a scalar and reach an `unreachable!()` inside `sample`.
#[test]
fn empty_data_vectors_are_rejected_by_the_graph_path() {
    let mut graph = Graph::new();
    let beta = Normal::prior(&mut graph, "beta", 0.0, 1.0);
    let x = graph.add_data("x", Vec::new());
    let mu = graph.scalar_mul_data(beta, x);
    let one = graph.add_constant(1.0);
    let obs = graph.add_obs_data(Vec::new());
    graph.normal_obs_logp(mu, one, obs);
    let error = sample(graph, SamplerConfig::default()).expect_err("empty data must not sample");
    assert!(error.contains("must not be empty"), "{error}");

    let mut graph = Graph::new();
    graph.add_data("x", Vec::new());
    assert!(DataBinding::from_graph(&graph).is_err());
}

/// `Graph::nodes` is public, so a node can name a later node or a parameter
/// that does not exist; shape validation indexed by those ids and panicked.
#[test]
fn out_of_order_and_dangling_references_are_rejected() {
    use rustmc_core::graph::{NodeId, Op};
    let mut graph = Graph::new();
    let x = Normal::prior(&mut graph, "x", 0.0, 1.0);
    let later = NodeId(graph.nodes.len() + 1);
    graph.add(x, later);
    graph.add_constant(1.0);
    let message = graph.validate_shapes().unwrap_err().to_string();
    assert!(message.contains("not an earlier node"), "{message}");
    assert!(sample(graph, SamplerConfig::default()).is_err());

    let mut graph = Graph::new();
    Normal::prior(&mut graph, "x", 0.0, 1.0);
    graph.nodes[0].op = Op::Param(5);
    let message = graph.validate_shapes().unwrap_err().to_string();
    assert!(message.contains("parameters 5..6"), "{message}");
    assert!(Evaluator::try_new(&graph).is_err());
}

/// The total log density sums node scalars, so a vector registered as a term,
/// or passed where a scalar is read, silently contributed zero or garbage.
#[test]
fn vector_nodes_in_scalar_positions_are_rejected() {
    let mut graph = Graph::new();
    let beta = Normal::prior(&mut graph, "beta", 0.0, 1.0);
    let x = graph.add_data("x", vec![1.0, 2.0, 3.0]);
    let mu = graph.scalar_mul_data(beta, x);
    graph.add_logp_term(mu);
    let message = graph.validate_shapes().unwrap_err().to_string();
    assert!(message.contains("log-density term"), "{message}");

    let mut graph = Graph::new();
    let beta = Normal::prior(&mut graph, "beta", 0.0, 1.0);
    let x = graph.add_data("x", vec![1.0, 2.0, 3.0]);
    let scale = graph.scalar_mul_data(beta, x);
    let zero = graph.add_constant(0.0);
    graph.normal_logp(beta, zero, scale);
    let message = graph.validate_shapes().unwrap_err().to_string();
    assert!(message.contains("as a scalar"), "{message}");
    assert!(sample(graph, SamplerConfig::default()).is_err());
}

/// The raw kernels returned `expect` panics for data that do not bind and
/// never checked `init` at all.
#[test]
fn raw_kernels_return_errors_for_bad_inputs() {
    use rand::SeedableRng;
    use rustmc_core::hmc::{self, HmcConfig};
    use rustmc_core::nuts::{self, NutsConfig};
    let mut graph = Graph::new();
    Normal::prior(&mut graph, "x", 0.0, 1.0);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let nuts_config = NutsConfig {
        num_draws: 2,
        num_warmup: 2,
        ..Default::default()
    };
    let hmc_config = HmcConfig {
        num_draws: 2,
        num_warmup: 2,
        ..Default::default()
    };
    for init in [vec![], vec![0.0, 0.0], vec![f64::NAN]] {
        let error = nuts::run_chain(&graph, &nuts_config, &mut rng, Some(init.clone()), None)
            .expect_err("wrong init must be refused");
        assert!(error.contains("init"), "{error}");
        assert!(hmc::run_chain(&graph, &hmc_config, &mut rng, Some(init), None).is_err());
    }
    let mut empty = Graph::new();
    let beta = Normal::prior(&mut empty, "beta", 0.0, 1.0);
    let x = empty.add_data("x", Vec::new());
    empty.scalar_mul_data(beta, x);
    assert!(nuts::run_chain(&empty, &nuts_config, &mut rng, None, None).is_err());
    assert!(hmc::run_chain(&empty, &hmc_config, &mut rng, None, None).is_err());
}

/// Counts heap allocations made by the current thread while enabled, so the
/// hot-path claims below are checked rather than asserted in comments.
mod allocation_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub struct Counting;

    thread_local! {
        static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
    }

    fn record() {
        COUNT.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record();
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    /// Allocations `f` makes on this thread.
    pub fn count<T>(f: impl FnOnce() -> T) -> (usize, T) {
        COUNT.with(|count| count.set(Some(0)));
        let value = f();
        let n = COUNT.with(|count| count.replace(None)).unwrap_or(0);
        (n, value)
    }
}

#[global_allocator]
static ALLOCATOR: allocation_counter::Counting = allocation_counter::Counting;

fn regression_graph() -> Graph {
    let mut graph = Graph::new();
    let intercept = Normal::prior(&mut graph, "intercept", 0.0, 1.0);
    let start = graph.add_vector_params("beta", 3);
    graph.vector_normal_logp(start, 3, 0.0, 1.0);
    let rows = 12;
    let x: Vec<f64> = (0..rows * 3).map(|i| (i as f64 * 0.37).sin()).collect();
    let matrix = graph.store_matrix(x, rows, 3);
    let mu = graph.mat_vec_mul(matrix, start, 3, Some(intercept));
    let one = graph.add_constant(1.0);
    let y: Vec<f64> = (0..rows).map(|i| (i as f64 * 0.11).cos()).collect();
    let obs = graph.add_obs_data(y);
    graph.normal_obs_logp(mu, one, obs);
    graph
}

#[test]
fn evaluator_passes_do_not_allocate() {
    let graph = regression_graph();
    let mut evaluator = Evaluator::new(&graph);
    let position = [0.1, -0.2, 0.3, 0.05];
    evaluator.compute(&graph, &position);
    let (allocations, ()) = allocation_counter::count(|| {
        for _ in 0..50 {
            evaluator.compute(&graph, &position);
            evaluator.forward(&graph, &position);
        }
    });
    assert_eq!(allocations, 0);
}

#[test]
fn nuts_transitions_allocate_only_the_retained_draw() {
    use rand::SeedableRng;
    use rustmc_core::nuts::{self, NutsConfig};
    let graph = regression_graph();
    let run = |draws: usize| {
        let config = NutsConfig {
            num_warmup: 100,
            num_draws: draws,
            ..Default::default()
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        allocation_counter::count(|| {
            nuts::run_chain(&graph, &config, &mut rng, None, None).unwrap()
        })
    };
    let (short, short_chain) = run(200);
    let (long, long_chain) = run(400);
    // The two runs share their first 300 transitions exactly.
    assert_eq!(short_chain.samples[..], long_chain.samples[..200]);
    let extra_steps: usize = long_chain.transitions[300..]
        .iter()
        .map(|t| t.num_leapfrog_steps)
        .sum();
    // Each extra draw clones its position once; the tree's phase points come
    // from a pool that has already grown to size. Before the pool, every
    // leapfrog step allocated nine vectors.
    let extra = long - short;
    assert!(
        extra <= 200 + 16,
        "{extra} allocations for 200 draws and {extra_steps} leapfrog steps"
    );
}
