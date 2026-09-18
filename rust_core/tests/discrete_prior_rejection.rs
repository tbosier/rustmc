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

use std::sync::Arc;

use rustmc_core::autodiff::Evaluator;
use rustmc_core::data::DataBinding;
use rustmc_core::distributions::{Bernoulli, Normal, Poisson};
use rustmc_core::graph::Graph;
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

/// The four `sampler` entry points a caller can reach directly. Named for what
/// it actually covers: the raw `nuts`/`hmc` `run_chain` kernels are `pub` too
/// and are deliberately not covered (they return a bare `ChainResult`, with no
/// channel to report a rejection on).
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
