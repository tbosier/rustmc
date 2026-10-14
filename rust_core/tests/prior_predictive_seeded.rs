//! `prior_predictive_seeded` is the generator Python's
//! `sample_prior_predictive` uses. It must be exactly `prior_predictive` on a
//! `ChaCha8Rng` keyed by `stream_seed(seed, PRIOR_PREDICT_SEED_DOMAIN)`, so
//! that a Rust caller reproduces Python's draws bit for bit.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::model::{compile, ModelSpec};
use rustmc_core::prior_sampling::{prior_predictive, prior_predictive_seeded};
use rustmc_core::seeding::{stream_seed, PRIOR_PREDICT_SEED_DOMAIN};
use std::collections::HashMap;

#[test]
fn seeded_prior_predictive_matches_the_manual_generator() {
    let definition: ModelSpec = serde_json::from_value(serde_json::json!({
        "dimensions": {}, "potentials": [], "deterministics": [],
        "priors": [
            {"Normal": {"name": "mu", "mu": {"Const": 0.0}, "sigma": {"Const": 1.0}}},
            {"HalfNormal": {"name": "sigma", "sigma": {"Const": 1.0}}}
        ],
        "likelihoods": [{
            "family": "Normal", "name": "obs", "mu_expr": {"Param": "mu"},
            "sigma": {"Param": "sigma"}, "observed_key": "y"
        }]
    }))
    .unwrap();
    let data = HashMap::from([("y".to_string(), vec![0.4, -0.2, 0.9])]);
    let compiled = compile(&definition, &data, &HashMap::new()).unwrap();

    for seed in [0u64, 42, u64::MAX] {
        let mut rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, PRIOR_PREDICT_SEED_DOMAIN));
        let manual = prior_predictive(
            &compiled.graph,
            &definition.priors,
            &compiled.display_params,
            &compiled.auto_vector_params,
            64,
            &mut rng,
        )
        .unwrap();
        let seeded = prior_predictive_seeded(
            &compiled.graph,
            &definition.priors,
            &compiled.display_params,
            &compiled.auto_vector_params,
            64,
            seed,
        )
        .unwrap();
        assert_eq!(manual, seeded, "seed {seed}");
        assert_eq!(seeded.predictions[0].len(), 64 * 3);
    }
}
