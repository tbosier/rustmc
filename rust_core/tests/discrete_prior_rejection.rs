//! Gradient-based sampling must refuse discrete latent parameters.
//!
//! `Bernoulli::prior` and `Poisson::prior` declare an unconstrained continuous
//! parameter carrying a density defined only on the integers. Unguarded, the
//! two fail differently and neither reports a support error: Bernoulli does not
//! check its support, so the chain random-walks and returns fractional draws;
//! Poisson does check (via `count_sampling::log_mass`), so every off-integer
//! proposal is rejected and the chain is pinned to its integer start, diverging
//! on every transition. The sampler now rejects both up front.
//!
//! Discrete *observations* are a different construction and must keep working,
//! as must prior-predictive evaluation — the negative controls below cover both.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::DataBinding;
use rustmc_core::distributions::{Bernoulli, Normal, Poisson};
use rustmc_core::graph::{ElementwiseOp, Graph};
use rustmc_core::hmc::{self, HmcConfig};
use rustmc_core::nuts::{self, NutsConfig};
use rustmc_core::sampler::{
    batch_sample_graphs, sample, sample_batch_bound, sample_bound, sample_bound_with_init,
    BatchSampleConfig, SamplerConfig,
};

fn config() -> SamplerConfig {
    SamplerConfig {
        num_chains: 1,
        num_draws: 20,
        num_warmup: 20,
        seed: 20260918,
        num_threads: 1,
        show_progress: false,
        ..Default::default()
    }
}

fn bind(graph: Graph) -> (Arc<Graph>, DataBinding) {
    let binding = DataBinding::from_graph(&graph).expect("graph binds");
    (Arc::new(graph.structure_only()), binding)
}

/// A continuous Normal observation model, plus one discrete latent built by
/// `prior_fn`. The Normal part alone would sample without complaint.
fn graph_with_discrete_latent(prior_fn: impl FnOnce(&mut Graph)) -> Graph {
    let mut graph = Graph::new();
    prior_fn(&mut graph);
    let mu = Normal::prior(&mut graph, "mu", 0.0, 5.0);
    let obs = graph.add_obs_data(vec![0.9, 1.1, 1.4, 0.7, 1.0]);
    let means = graph.broadcast_observation(mu, obs);
    let sigma = graph.add_constant(1.0);
    graph.normal_obs_logp(means, sigma, obs);
    graph
}

#[test]
fn bernoulli_prior_is_rejected_by_sample_bound() {
    let graph = graph_with_discrete_latent(|g| {
        Bernoulli::prior(g, "flag", 0.5);
    });
    let (structure, binding) = bind(graph);
    let error = sample_bound(structure, binding, config())
        .expect_err("a Bernoulli latent must not be sampled with NUTS");
    // Match the offender entry itself, not the loose word "Bernoulli": the
    // explanatory tail of the message names both families every time, so a
    // bare `contains("Bernoulli")` would also pass for a Poisson offender.
    assert!(
        error.contains("'flag' (Bernoulli)"),
        "error should name the offending parameter and its family: {error}"
    );
    assert!(
        error.contains("discrete"),
        "error should say why it is refused: {error}"
    );
}

#[test]
fn poisson_prior_is_rejected_by_sample_bound() {
    let graph = graph_with_discrete_latent(|g| {
        Poisson::prior(g, "events", 3.0);
    });
    let (structure, binding) = bind(graph);
    let error = sample_bound(structure, binding, config())
        .expect_err("a Poisson latent must not be sampled with NUTS");
    assert!(
        error.contains("'events' (Poisson)"),
        "error should name the offending parameter and its family: {error}"
    );
    // The offender list must not pick up the unrelated continuous parameter.
    assert!(
        !error.contains("'mu'"),
        "only the discrete parameter is an offender: {error}"
    );
}

/// The four `sampler` entry points a caller can reach directly. The raw
/// `nuts`/`hmc` kernels are covered separately, below.
#[test]
fn each_sampler_entry_point_rejects_a_discrete_latent() {
    let build = || {
        graph_with_discrete_latent(|g| {
            Bernoulli::prior(g, "flag", 0.5);
        })
    };

    // Data-owning convenience wrapper.
    assert!(sample(build(), config())
        .expect_err("sample must reject")
        .contains("'flag'"));

    // Explicit-initialization variant, with a nonzero start so the rejection
    // cannot be mistaken for an initialization failure.
    let (structure, binding) = bind(build());
    let init = vec![vec![0.25, 0.5]];
    assert!(
        sample_bound_with_init(structure, binding, config(), Some(init))
            .expect_err("sample_bound_with_init must reject")
            .contains("'flag'")
    );

    // The independent-graph batch path drives `run_chain` directly and so does
    // not pass through `sample_bound_with_init`.
    let batch = BatchSampleConfig {
        num_chains: 1,
        num_draws: 20,
        num_warmup: 20,
        seed: 20260918,
        show_progress: false,
        ..Default::default()
    };
    assert!(batch_sample_graphs(vec![build()], batch.clone())
        .expect_err("batch_sample_graphs must reject")
        .contains("'flag'"));

    // The shared-structure batch path, which re-enters `sample_bound` per cell.
    let (structure, binding) = bind(build());
    let error = sample_batch_bound(structure, vec![binding], batch)
        .expect_err("sample_batch_bound must reject");
    assert!(error.contains("'flag'"), "{error}");
}

/// `nuts::run_chain`, `nuts::run_chain_bound`, `hmc::run_chain` and
/// `hmc::run_chain_bound` are `pub` and drive the samplers without passing
/// through `sampler`, so until they carried an error channel a caller could
/// reach past every check above and run a discrete latent anyway.
#[test]
fn the_raw_kernels_reject_a_discrete_latent() {
    let build = || {
        graph_with_discrete_latent(|g| {
            Bernoulli::prior(g, "flag", 0.5);
        })
    };
    let nuts_config = NutsConfig {
        num_draws: 20,
        num_warmup: 20,
        ..Default::default()
    };
    let hmc_config = HmcConfig {
        num_draws: 20,
        num_warmup: 20,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(20260918);

    let graph = build();
    assert!(nuts::run_chain(&graph, &nuts_config, &mut rng, None, None)
        .expect_err("nuts::run_chain must reject")
        .contains("'flag'"));
    assert!(hmc::run_chain(&graph, &hmc_config, &mut rng, None, None)
        .expect_err("hmc::run_chain must reject")
        .contains("'flag'"));

    let binding = DataBinding::from_graph(&graph).expect("graph binds");
    assert!(
        nuts::run_chain_bound(&graph, binding.clone(), &nuts_config, &mut rng, None, None)
            .expect_err("nuts::run_chain_bound must reject")
            .contains("'flag'")
    );
    assert!(
        hmc::run_chain_bound(&graph, binding, &hmc_config, &mut rng, None, None)
            .expect_err("hmc::run_chain_bound must reject")
            .contains("'flag'")
    );
}

/// Negative control for the same four: a continuous model still runs through
/// them, so the error channel did not cost the kernels their purpose.
#[test]
fn the_raw_kernels_still_run_a_continuous_model() {
    let graph = graph_with_discrete_latent(|_| {});
    let nuts_config = NutsConfig {
        num_draws: 20,
        num_warmup: 20,
        ..Default::default()
    };
    let hmc_config = HmcConfig {
        num_draws: 20,
        num_warmup: 20,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(20260918);

    let chain = nuts::run_chain(&graph, &nuts_config, &mut rng, None, None)
        .expect("nuts::run_chain must still run a continuous model");
    assert_eq!(chain.samples.len(), 20);

    let chain = hmc::run_chain(&graph, &hmc_config, &mut rng, None, None)
        .expect("hmc::run_chain must still run a continuous model");
    assert_eq!(chain.samples.len(), 20);

    let binding = DataBinding::from_graph(&graph).expect("graph binds");
    let chain = nuts::run_chain_bound(&graph, binding.clone(), &nuts_config, &mut rng, None, None)
        .expect("nuts::run_chain_bound must still run a continuous model");
    assert_eq!(chain.samples.len(), 20);

    let chain = hmc::run_chain_bound(&graph, binding, &hmc_config, &mut rng, None, None)
        .expect("hmc::run_chain_bound must still run a continuous model");
    assert_eq!(chain.samples.len(), 20);
}

/// A free parameter that reaches the discrete density through another node is
/// the same invalid model, and worse in one respect: `Op::BernoulliLogP`'s
/// backward pass propagates no adjoint to `x` at all, so the term moves the
/// density without moving the gradient. Nothing in this crate builds these
/// shapes — they are what the published `Graph` API lets a caller build.
#[test]
fn a_discrete_density_over_a_transformed_parameter_is_rejected() {
    type Wire = fn(&mut Graph);
    let shapes: [(&str, Wire); 6] = [
        ("exp", |g| {
            let raw = g.add_param("flag");
            let x = g.exp(raw);
            let p = g.add_constant(0.5);
            g.bernoulli_logp(x, p);
        }),
        ("sigmoid", |g| {
            let raw = g.add_param("flag");
            let x = g.sigmoid(raw);
            let p = g.add_constant(0.5);
            g.bernoulli_logp(x, p);
        }),
        ("bounded sigmoid", |g| {
            let raw = g.add_param("flag");
            let x = g.bounded_sigmoid(raw, 0.0, 1.0);
            let p = g.add_constant(0.5);
            g.bernoulli_logp(x, p);
        }),
        // Two nodes deep, so a one-level check would not be enough either.
        ("add then multiply", |g| {
            let raw = g.add_param("flag");
            let one = g.add_constant(1.0);
            let shifted = g.add(raw, one);
            let x = g.mul(shifted, one);
            let p = g.add_constant(0.5);
            g.bernoulli_logp(x, p);
        }),
        ("elementwise", |g| {
            let raw = g.add_param("flag");
            let two = g.add_constant(2.0);
            let x = g.elementwise(ElementwiseOp::Mul, raw, Some(two));
            let p = g.add_constant(0.5);
            g.bernoulli_logp(x, p);
        }),
        // Poisson reaches its rate the same way.
        ("poisson over exp", |g| {
            let raw = g.add_param("flag");
            let x = g.exp(raw);
            let lam = g.add_constant(3.0);
            g.poisson_logp(x, lam);
        }),
    ];

    for (shape, wire) in shapes {
        let graph = graph_with_discrete_latent(wire);
        let (structure, binding) = bind(graph);
        let Err(error) = sample_bound(structure, binding, config()) else {
            panic!("{shape}: an indirect discrete latent must not be sampled with NUTS");
        };
        assert!(
            error.contains("'flag'"),
            "{shape}: error should name the offending parameter: {error}"
        );
        assert!(
            !error.contains("'mu'"),
            "{shape}: only the discrete parameter is an offender: {error}"
        );
    }
}

/// Negative control: the walk must not start rejecting models because some
/// parameter happens to be reachable from a *continuous* term. Only the
/// discrete densities are visited at all.
#[test]
fn a_transformed_parameter_under_a_continuous_density_still_samples() {
    let mut graph = Graph::new();
    let raw = graph.add_param("theta");
    let scale = graph.exp(raw);
    let zero = graph.add_constant(0.0);
    let one = graph.add_constant(1.0);
    graph.normal_logp(raw, zero, one);
    graph.normal_logp(zero, zero, scale);

    let (structure, binding) = bind(graph);
    let result = sample_bound(structure, binding, config())
        .expect("a continuous density over a transformed parameter must still fit");
    assert_eq!(result.param_names, vec!["theta".to_string()]);
}

/// Negative control: a discrete density whose `x` depends on no free parameter
/// is a constant term, not a latent. It must not be rejected.
#[test]
fn a_discrete_density_over_a_constant_is_not_a_latent() {
    let graph = graph_with_discrete_latent(|g| {
        let x = g.add_constant(1.0);
        let p = g.add_constant(0.5);
        g.bernoulli_logp(x, p);
    });
    let (structure, binding) = bind(graph);
    sample_bound(structure, binding, config())
        .expect("a Bernoulli term over a constant has no discrete latent to refuse");
}

/// Negative control: a Bernoulli-logit likelihood over observed 0/1 data has no
/// discrete latent — the sampled parameter is the continuous linear predictor.
#[test]
fn bernoulli_logit_likelihood_on_observed_data_still_samples() {
    let mut graph = Graph::new();
    let eta = Normal::prior(&mut graph, "eta", 0.0, 2.0);
    let obs = graph.add_obs_data(vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    let linpred = graph.broadcast_observation(eta, obs);
    graph.obs_logp_bernoulli_logit(linpred, obs);

    let (structure, binding) = bind(graph);
    let result = sample_bound(structure, binding, config()).expect("beta-Bernoulli must still fit");
    assert_eq!(result.param_names, vec!["eta".to_string()]);
    assert!(result.samples[0].iter().all(|draw| draw[0].is_finite()));
}

/// Negative control: a Poisson-log likelihood over observed counts, likewise.
#[test]
fn poisson_log_likelihood_on_observed_data_still_samples() {
    let mut graph = Graph::new();
    let eta = Normal::prior(&mut graph, "eta", 0.0, 2.0);
    let obs = graph.add_obs_data(vec![3.0, 5.0, 2.0, 4.0, 6.0, 3.0]);
    let linpred = graph.broadcast_observation(eta, obs);
    graph.obs_logp_poisson_log(linpred, obs);

    let (structure, binding) = bind(graph);
    let result = sample_bound(structure, binding, config()).expect("gamma-Poisson must still fit");
    assert_eq!(result.param_names, vec!["eta".to_string()]);
    assert!(result.samples[0].iter().all(|draw| draw[0].is_finite()));
}

/// Negative control: the rejection is scoped to gradient-based *sampling*. A
/// Bernoulli prior must still evaluate, because prior-predictive simulation
/// drives the evaluator directly and never calls the sampler.
#[test]
fn discrete_prior_still_evaluates_for_prior_predictive_use() {
    let graph = graph_with_discrete_latent(|g| {
        Bernoulli::prior(g, "flag", 0.5);
    });
    let mut evaluator = Evaluator::new(&graph);

    // logp(1 | p=0.5) = ln 0.5, and the density is defined at both support points.
    for flag in [0.0, 1.0] {
        evaluator.compute(&graph, &[flag, 1.0]);
        assert!(
            evaluator.total_logp.is_finite(),
            "prior-predictive evaluation must stay finite at x={flag}"
        );
    }

    // Sampling the very same graph is still refused.
    let (structure, binding) = bind(graph);
    assert!(sample_bound(structure, binding, config()).is_err());
}

/// Negative control: a purely continuous model is untouched by the scan.
#[test]
fn continuous_model_is_unaffected() {
    let graph = graph_with_discrete_latent(|_| {});
    let (structure, binding) = bind(graph);
    sample_bound(structure, binding, config()).expect("continuous model must still fit");
}
