//! A discrete prior must survive loading, and be refused only at sampling.
//!
//! `Bernoulli::prior` and `Poisson::prior` are deliberately kept "available for
//! prior-predictive simulation" while gradient sampling refuses them -- the
//! intent is documented on both constructors in `distributions.rs` and on
//! `reject_discrete_priors_for_gradient_sampling` in the binding crate.
//!
//! `GraphModel::from_artifact` used to apply that rejection at *load* time, so
//! an artifact carrying a Bernoulli prior could not be opened at all and the
//! simulation it was kept for was unreachable. The rejection belongs at the
//! sampling chokepoint, where `sampler::reject_discrete_latent_parameters`
//! already enforces it on the compiled graph; the tests below pin down both
//! halves of that -- the load now succeeds, and sampling still does not.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::{DisplayParamSpec, GraphModel};
use rustmc_core::sampler::SamplerConfig;
use std::sync::Arc;

const ARTIFACT: &str = include_str!("fixtures/graph_model_v1.json");
const ROWS: usize = 6;

/// The shipped fixture with its potential dropped (a potential has no random
/// generator, so its prior is not simulable) and one discrete prior spliced in.
/// Everything else -- including the schema the loader cross-checks against the
/// recompiled definition -- is untouched, because a prior adds a parameter and
/// no data slot.
fn artifact_with_discrete_prior(prior: serde_json::Value) -> String {
    let mut artifact: serde_json::Value = serde_json::from_str(ARTIFACT).unwrap();
    artifact["definition"]["potentials"] = serde_json::json!([]);
    artifact["definition"]["priors"]
        .as_array_mut()
        .unwrap()
        .push(prior);
    artifact.to_string()
}

fn bernoulli_artifact() -> String {
    artifact_with_discrete_prior(serde_json::json!({
        "Bernoulli": {"name": "flag", "p": 0.5}
    }))
}

fn poisson_artifact() -> String {
    artifact_with_discrete_prior(serde_json::json!({
        "Poisson": {"name": "events", "lam": 3.0}
    }))
}

fn inputs() -> DataInputs {
    let mut data = DataInputs::default();
    data.matrices.insert(
        "X".into(),
        MatrixBinding {
            data: Arc::from(
                (0..ROWS)
                    .flat_map(|i| [1., i as f64 / 3.])
                    .collect::<Vec<_>>(),
            ),
            n_rows: ROWS,
            n_cols: 2,
        },
    );
    data.vectors.insert(
        "y".into(),
        Arc::from((0..ROWS).map(|i| i as f64 / 2.).collect::<Vec<_>>()),
    );
    data
}

fn display_names(model: &GraphModel) -> Vec<String> {
    model
        .display_params
        .iter()
        .map(|spec| match spec {
            DisplayParamSpec::Raw { name, .. }
            | DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name.clone(),
        })
        .collect()
}

fn column<'a>(model: &GraphModel, draws: &'a [Vec<f64>], name: &str) -> &'a [f64] {
    let index = display_names(model)
        .iter()
        .position(|candidate| candidate == name)
        .unwrap_or_else(|| panic!("no display parameter named {name}"));
    &draws[index]
}

#[test]
fn an_artifact_carrying_a_bernoulli_prior_loads() {
    let model = GraphModel::from_json(&bernoulli_artifact())
        .expect("a discrete prior must not block loading the artifact it lives in");
    assert!(display_names(&model).contains(&"flag".to_string()));
}

#[test]
fn an_artifact_carrying_a_poisson_prior_loads() {
    GraphModel::from_json(&poisson_artifact())
        .expect("a discrete prior must not block loading the artifact it lives in");
}

#[test]
fn a_loaded_bernoulli_prior_simulates_its_prior_predictive() {
    let model = GraphModel::from_json(&bernoulli_artifact()).unwrap();
    let binding = model.bind(inputs(), "prior").unwrap();
    let n_samples = 2000;
    let mut rng = ChaCha8Rng::seed_from_u64(20260918);
    let draws = model
        .prior_predictive(&binding, n_samples, &mut rng)
        .expect("a discrete prior is kept for exactly this");

    let flag = column(&model, &draws.params, "flag");
    assert_eq!(flag.len(), n_samples);
    assert!(
        flag.iter().all(|x| *x == 0.0 || *x == 1.0),
        "a Bernoulli draw must land on its support"
    );
    // p = 0.5, so a two-sided 5-sigma band on 2000 draws is +/- 0.056.
    let mean = flag.iter().sum::<f64>() / n_samples as f64;
    assert!((mean - 0.5).abs() < 0.056, "Bernoulli(0.5) mean was {mean}");

    // The continuous part of the same model is unaffected.
    assert!(draws.predictions[0].iter().all(|y| y.is_finite()));
}

#[test]
fn a_loaded_poisson_prior_simulates_its_prior_predictive() {
    let model = GraphModel::from_json(&poisson_artifact()).unwrap();
    let binding = model.bind(inputs(), "prior").unwrap();
    let n_samples = 2000;
    let mut rng = ChaCha8Rng::seed_from_u64(20260919);
    let draws = model
        .prior_predictive(&binding, n_samples, &mut rng)
        .expect("a discrete prior is kept for exactly this");

    let events = column(&model, &draws.params, "events");
    assert!(
        events.iter().all(|x| *x >= 0.0 && x.fract() == 0.0),
        "a Poisson draw must be a non-negative integer"
    );
    // lambda = 3, variance 3: a 5-sigma band on 2000 draws is +/- 0.194.
    let mean = events.iter().sum::<f64>() / n_samples as f64;
    assert!((mean - 3.0).abs() < 0.194, "Poisson(3) mean was {mean}");
}

#[test]
fn a_single_prior_draw_from_a_loaded_discrete_model_works() {
    let model = GraphModel::from_json(&bernoulli_artifact()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let draw = model.sample_prior(&mut rng).expect("prior draw");
    assert_eq!(draw.display.len(), display_names(&model).len());
}

/// The load-time check is gone; this is what must still stop a user.
#[test]
fn sampling_a_loaded_discrete_model_is_still_refused() {
    for (artifact, offender) in [
        (bernoulli_artifact(), "'flag' (Bernoulli)"),
        (poisson_artifact(), "'events' (Poisson)"),
    ] {
        let model = GraphModel::from_json(&artifact).unwrap();
        let binding = model.bind(inputs(), "fit").unwrap();
        let config = SamplerConfig {
            num_chains: 1,
            num_draws: 10,
            num_warmup: 10,
            seed: 20260918,
            num_threads: 1,
            show_progress: false,
            ..Default::default()
        };
        let error = model
            .sample(binding, config, None)
            .expect_err("gradient sampling must still refuse a discrete latent")
            .to_string();
        assert!(
            error.contains(offender),
            "the rejection must name the offending parameter and family: {error}"
        );
        assert!(
            error.contains("discrete"),
            "the rejection must say why: {error}"
        );
    }
}

/// `log_density` is not sampling and never was refused; it stays available, so
/// a caller can still evaluate the target of a model it loaded.
#[test]
fn log_density_of_a_loaded_discrete_model_is_available() {
    let model = GraphModel::from_json(&bernoulli_artifact()).unwrap();
    let binding = model.bind(inputs(), "eval").unwrap();
    let mut position = vec![0.0; model.structure.param_count];
    // `flag` is the last declared prior, so it is the last parameter.
    *position.last_mut().unwrap() = 1.0;
    let (logp, grad) = model.log_density(&binding, &position).expect("evaluates");
    assert!(logp.is_finite(), "log density at a support point: {logp}");
    assert_eq!(grad.len(), position.len());
}
