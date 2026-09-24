//! A request too large to hold must come back as an error. A Rust allocation
//! that fails aborts the process, so without these checks an oversized
//! `draws`, prior-predictive sample count or prediction size took the Python
//! interpreter down with it.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::data::DataInputs;
use rustmc_core::distributions::Normal;
use rustmc_core::graph::Graph;
use rustmc_core::model::{compile, GraphModel, ModelSpec};
use rustmc_core::sampler::{batch_sample_graphs, sample, BatchSampleConfig, SamplerConfig};
use std::collections::HashMap;
use std::sync::Arc;

const HUGE: usize = 1_000_000_000_000;

fn intercept_model() -> GraphModel {
    let definition: ModelSpec = serde_json::from_value(serde_json::json!({
        "dimensions": {}, "potentials": [], "deterministics": [],
        "priors": [{"Normal": {"name": "mu", "mu": {"Const": 0.0}, "sigma": {"Const": 1.0}}}],
        "likelihoods": [{
            "family": "Normal", "name": "obs", "mu_expr": {"Param": "mu"},
            "sigma": {"Const": 1.0}, "observed_key": "y"
        }]
    }))
    .unwrap();
    let data = HashMap::from([("y".to_string(), vec![0.4, -0.2, 0.9])]);
    let compiled = compile(&definition, &data, &HashMap::new()).unwrap();
    GraphModel {
        definition,
        structure: Arc::new(compiled.graph.structure_only()),
        likelihood_names: compiled.likelihood_names,
        display_params: compiled.display_params,
    }
}

fn observed() -> DataInputs {
    DataInputs {
        vectors: HashMap::from([("y".to_string(), Arc::from(vec![0.4, -0.2, 0.9]))]),
        ..DataInputs::default()
    }
}

fn quiet(config: SamplerConfig) -> SamplerConfig {
    SamplerConfig {
        show_progress: false,
        ..config
    }
}

fn assert_refused<T: std::fmt::Debug>(result: Result<T, impl std::fmt::Display>) {
    let error = result.unwrap_err().to_string();
    assert!(error.contains("safety limit"), "{error}");
}

fn scalar_graph() -> Graph {
    let mut graph = Graph::new();
    Normal::prior(&mut graph, "x", 0.0, 1.0);
    graph
}

#[test]
fn oversized_fits_are_refused_before_sampling() {
    for config in [
        SamplerConfig {
            num_draws: HUGE,
            ..Default::default()
        },
        SamplerConfig {
            num_warmup: HUGE,
            ..Default::default()
        },
        SamplerConfig {
            num_chains: HUGE,
            ..Default::default()
        },
    ] {
        assert_refused(sample(scalar_graph(), quiet(config)));
    }

    // Default draws of a model with 300,000 parameters: 1.2 billion values.
    let mut wide = Graph::new();
    let start = wide.add_vector_params("b", 300_000);
    wide.vector_normal_logp(start, 300_000, 0.0, 1.0);
    assert_refused(sample(wide.clone(), quiet(SamplerConfig::default())));
    let batch = BatchSampleConfig {
        num_chains: 4,
        num_draws: 1000,
        show_progress: false,
        ..Default::default()
    };
    assert_refused(batch_sample_graphs(vec![wide], batch.clone()));
    assert_refused(batch_sample_graphs(
        vec![scalar_graph()],
        BatchSampleConfig {
            num_draws: HUGE,
            ..batch
        },
    ));

    let model = intercept_model();
    let binding = model.bind(observed(), "fit").unwrap();
    let config = quiet(SamplerConfig {
        num_draws: HUGE,
        ..Default::default()
    });
    assert_refused(model.sample(binding, config, None));
}

#[test]
fn oversized_prior_predictive_requests_are_refused() {
    let model = intercept_model();
    let binding = model.bind(observed(), "prior").unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    assert_refused(model.prior_predictive(&binding, HUGE, &mut rng));
    // A request of ordinary size still runs.
    assert!(model.prior_predictive(&binding, 10, &mut rng).is_ok());
}

#[test]
fn oversized_predictions_are_refused() {
    let model = intercept_model();
    let binding = model.bind(observed(), "fit").unwrap();
    let config = quiet(SamplerConfig {
        num_chains: 2,
        num_draws: 20,
        num_warmup: 50,
        ..Default::default()
    });
    let fit = model.sample(binding, config, None).unwrap();
    for size in [HUGE, usize::MAX] {
        let sizes = HashMap::from([("obs".to_string(), size)]);
        assert_refused(fit.predict(DataInputs::default(), sizes, 1, false));
    }
    let sizes = HashMap::from([("obs".to_string(), 5usize)]);
    let prediction = fit.predict(DataInputs::default(), sizes, 1, false).unwrap();
    assert_eq!(prediction["obs"][1][19].len(), 5);
}
