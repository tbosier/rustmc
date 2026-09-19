//! The native posterior-predictive stream must not be one of the fitting
//! chains' streams.
//!
//! `sampler::sample` seeds chain `i` with `config.seed.wrapping_add(i)`, so a
//! prediction that seeded `ChaCha8Rng::seed_from_u64(seed)` directly replayed
//! chain zero's draws whenever the caller passed the fit seed — which is the
//! natural thing for a caller to do. The oracle below is `rand_chacha` and
//! `rand_distr` used directly, not another copy of the crate's own sampling.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rustmc_core::data::DataInputs;
use rustmc_core::model::{
    compile, stream_seed, GraphModel, ModelSpec, POSTERIOR_PREDICT_SEED_DOMAIN,
    PRIOR_PREDICT_SEED_DOMAIN,
};
use rustmc_core::sampler::SamplerConfig;
use std::collections::HashMap;
use std::sync::Arc;

/// `SamplerConfig::default().seed`, and the seed a caller reaches for again
/// when predicting.
const SEED: u64 = 42;

const OBSERVATIONS: [f64; 3] = [0.4, -0.2, 0.9];

/// An intercept-only Normal model: the linear predictor *is* the sampled
/// parameter and the likelihood's sigma is exactly one, so a predictive draw is
/// `mu + z` for a single standard normal `z` off the prediction RNG.
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
    let data = HashMap::from([("y".to_string(), OBSERVATIONS.to_vec())]);
    let compiled = compile(&definition, &data, &HashMap::new()).unwrap();
    GraphModel {
        definition,
        structure: Arc::new(compiled.graph.structure_only()),
        likelihood_names: compiled.likelihood_names,
        display_params: compiled.display_params,
    }
}

fn observed_inputs() -> DataInputs {
    DataInputs {
        vectors: HashMap::from([("y".to_string(), Arc::from(OBSERVATIONS.to_vec()))]),
        ..DataInputs::default()
    }
}

#[test]
fn native_prediction_does_not_replay_the_fitting_chain_stream() {
    let model = intercept_model();
    let binding = model.bind(observed_inputs(), "fit").unwrap();
    let fit = model
        .sample(
            binding,
            SamplerConfig {
                num_chains: 2,
                num_draws: 8,
                num_warmup: 60,
                show_progress: false,
                seed: SEED,
                ..Default::default()
            },
            None,
        )
        .unwrap();

    let sizes = HashMap::from([("obs".to_string(), 1usize)]);
    let sampled = fit
        .predict(DataInputs::default(), sizes.clone(), SEED, false)
        .unwrap();
    let expected = fit
        .predict(DataInputs::default(), sizes, SEED, true)
        .unwrap();

    // With `expected`, the prediction is the linear predictor itself, which
    // confirms that the mean the sampled draw is centred on is this parameter.
    let mu = expected["obs"][0][0][0];
    assert_eq!(mu, fit.samples.samples[0][0][0]);

    // So the first sampled prediction is `Normal(mu, 1).sample(rng)` off the
    // prediction RNG's very first draw. It must not be the draw a generator
    // seeded with the fit seed hands out, because that is chain zero's.
    let mut replay = ChaCha8Rng::seed_from_u64(SEED);
    let replayed = Normal::new(mu, 1.0).unwrap().sample(&mut replay);
    assert_ne!(
        sampled["obs"][0][0][0], replayed,
        "prediction seeded with the fit seed replays chain zero's stream"
    );

    // The prediction is still reproducible, and a different seed still gives a
    // different answer.
    assert_eq!(
        sampled,
        fit.predict(
            DataInputs::default(),
            HashMap::from([("obs".to_string(), 1usize)]),
            SEED,
            false
        )
        .unwrap()
    );
    assert_ne!(
        sampled,
        fit.predict(
            DataInputs::default(),
            HashMap::from([("obs".to_string(), 1usize)]),
            SEED + 1,
            false
        )
        .unwrap()
    );
}

/// A simulation stream must not be any fitting chain's stream, and the two
/// simulation domains must not be each other's.
///
/// `sample` seeds chain `i` with `config.seed + i`, so for the collision to be
/// gone the re-keyed seed has to miss every chain index a run could plausibly
/// use, not just zero. Re-keying is also a bijection within a domain, so two
/// distinct seeds never land on the same stream.
#[test]
fn stream_seeds_avoid_the_fitting_chain_seeds_and_each_other() {
    for seed in [0u64, 1, 42, 7, 999, u64::MAX, u64::MAX - 3, 1 << 40] {
        let posterior = stream_seed(seed, POSTERIOR_PREDICT_SEED_DOMAIN);
        let prior = stream_seed(seed, PRIOR_PREDICT_SEED_DOMAIN);
        assert_ne!(posterior, prior, "domains collide at {seed}");
        for chain in 0..4096u64 {
            assert_ne!(
                posterior,
                seed.wrapping_add(chain),
                "posterior stream at {seed} is chain {chain}'s"
            );
            assert_ne!(
                prior,
                seed.wrapping_add(chain),
                "prior stream at {seed} is chain {chain}'s"
            );
        }
    }
    let mut seen = std::collections::HashSet::new();
    for seed in 0..10_000u64 {
        assert!(
            seen.insert(stream_seed(seed, POSTERIOR_PREDICT_SEED_DOMAIN)),
            "two seeds share a posterior stream at {seed}"
        );
    }
}
