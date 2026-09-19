//! The native posterior-predictive stream must not be the stream a fitting
//! chain used for the seed the caller actually passed.
//!
//! `sampler::sample` seeds chain `i` with `config.seed.wrapping_add(i)`, so a
//! prediction that seeded `ChaCha8Rng::seed_from_u64(seed)` directly replayed
//! chain zero's draws whenever the caller passed the fit seed — which is the
//! natural thing for a caller to do. The oracle below is `rand_chacha` and
//! `rand_distr` used directly, not another copy of the crate's own sampling.
//!
//! Re-keying is a permutation of the 64-bit seeds, not a partition of them, so
//! it cannot make the two spaces disjoint: for any fit seed there exists some
//! prediction seed that maps onto it. What it removes is the systematic
//! coincidence — the one a caller reaches by passing the same integer twice.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rustmc_core::data::DataInputs;
use rustmc_core::model::{compile, GraphModel, ModelSpec};
use rustmc_core::sampler::SamplerConfig;
use rustmc_core::seeding::{stream_seed, POSTERIOR_PREDICT_SEED_DOMAIN, PRIOR_PREDICT_SEED_DOMAIN};
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

/// For a seed a caller would pass, a simulation stream is not the stream any
/// plausible fitting chain used, and the two simulation domains differ.
///
/// `sample` seeds chain `i` with `config.seed + i`, so it is not enough for the
/// re-keyed seed to miss chain zero: it has to miss every chain index a run
/// could use. This checks the first 4096. It is a per-seed property, not a
/// universal one — re-keying permutes the seed space rather than partitioning
/// it — so the claim is about the seeds listed, and about the shape of the
/// mapping, not about every seed in `u64`.
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

/// The measurement behind the claim that separating the streams does not
/// correct an observable error, kept so the claim is reproducible.
///
/// Ignored because it fits four chains of a thousand draws. Run it with
/// `cargo test --release --test prediction_streams -- --ignored --nocapture`.
/// It prints, for the fit seed and three unrelated seeds, the standard
/// deviation of the standardised predictive residual `(y - a) / sigma` and its
/// correlation with each fitted parameter. On the pre-change seeding the fit
/// seed's row was not distinguishable from the others:
///
/// ```text
/// pred seed   resid sd   corr(resid,a)   corr(resid,sigma)
///        42     0.9863         -0.0321             -0.0089   <- fit chain 0
///        43     0.9993         -0.0161             -0.0018   <- fit chain 1
///         7     1.0031          0.0010             -0.0060
///       999     0.9869         -0.0004              0.0029
/// ```
///
/// With four chains the fit consumed seeds 42 through 45, so 43 is chain
/// one's stream and is a second collision rather than a control; 7 and 999
/// are the controls.
#[test]
#[ignore = "fits 4 x 2000 draws; run explicitly with --ignored"]
fn report_prediction_coupling_at_realistic_settings() {
    let definition: ModelSpec = serde_json::from_value(serde_json::json!({
        "dimensions": {}, "potentials": [], "deterministics": [],
        "priors": [
            {"Normal": {"name": "a", "mu": {"Const": 0.0}, "sigma": {"Const": 5.0}}},
            {"HalfNormal": {"name": "sigma", "sigma": {"Const": 2.0}}}
        ],
        "likelihoods": [{
            "family": "Normal", "name": "obs", "mu_expr": {"Param": "a"},
            "sigma": {"Param": "sigma"}, "observed_key": "y"
        }]
    }))
    .unwrap();
    let y: Vec<f64> = (0..40)
        .map(|i| {
            let x = (i as f64 + 0.5) / 40.0;
            1.5 + (2.0 * std::f64::consts::PI * x).sin() + 0.3 * (7.0 * x).cos()
        })
        .collect();
    let compiled = compile(
        &definition,
        &HashMap::from([("y".to_string(), y.clone())]),
        &HashMap::new(),
    )
    .unwrap();
    let model = GraphModel {
        definition,
        structure: Arc::new(compiled.graph.structure_only()),
        likelihood_names: compiled.likelihood_names,
        display_params: compiled.display_params,
    };
    let binding = model
        .bind(
            DataInputs {
                vectors: HashMap::from([("y".to_string(), Arc::from(y))]),
                ..DataInputs::default()
            },
            "fit",
        )
        .unwrap();
    let fit = model
        .sample(
            binding,
            SamplerConfig {
                num_chains: 4,
                num_draws: 1000,
                num_warmup: 1000,
                show_progress: false,
                seed: SEED,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let names = &fit.samples.param_names;
    let ia = names.iter().position(|n| n == "a").unwrap();
    let is = names.iter().position(|n| n == "sigma").unwrap();

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma) * (x - ma)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb) * (y - mb)).sum();
        cov / (va.sqrt() * vb.sqrt())
    }

    println!("pred seed   resid sd   corr(resid,a)   corr(resid,sigma)");
    for seed in [SEED, 43, 7, 999] {
        let pred = fit
            .predict(
                DataInputs::default(),
                HashMap::from([("obs".to_string(), 1usize)]),
                seed,
                false,
            )
            .unwrap();
        let (mut resid, mut avals, mut svals) = (Vec::new(), Vec::new(), Vec::new());
        for (ci, chain) in fit.samples.samples.iter().enumerate() {
            for (di, draw) in chain.iter().enumerate() {
                resid.push((pred["obs"][ci][di][0] - draw[ia]) / draw[is]);
                avals.push(draw[ia]);
                svals.push(draw[is]);
            }
        }
        let n = resid.len() as f64;
        let m = resid.iter().sum::<f64>() / n;
        let sd = (resid.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / (n - 1.0)).sqrt();
        println!(
            "{seed:>9}   {sd:>8.4}   {:>13.4}   {:>17.4}",
            corr(&resid, &avals),
            corr(&resid, &svals)
        );
        assert!(sd.is_finite() && (0.5..2.0).contains(&sd), "{sd}");
    }
}
