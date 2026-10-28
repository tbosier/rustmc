//! Every way a fitted model reads node values, on a grouped Normal model
//! whose fused chain does not write its vectors: posterior predictive draws
//! and expected values, predictions at new rows, pointwise log-likelihoods,
//! deterministics and prior predictive draws. Each is computed twice from
//! the same fit, once with grouped fusion allowed and once with it turned
//! off, and the two must agree to the bit. The deterministic and the
//! log-likelihood are also checked against values computed from the draws by
//! hand.
//!
//! Fusion is switched for the whole process, so this file holds one test.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::autodiff::{fused_grouped_normal_passes, set_grouped_normal_fusion};
use rustmc_core::data::DataInputs;
use rustmc_core::graph::ElementwiseOp;
use rustmc_core::model::{
    compile, GraphModel, HyperParam, LikelihoodFamily, LikelihoodSpec, ModelFit, ModelSpec, MuExpr,
    PriorSpec, SigmaSpec,
};
use rustmc_core::sampler::SamplerConfig;
use std::collections::HashMap;
use std::sync::Arc;

const GROUPS: usize = 6;

fn mean() -> MuExpr {
    MuExpr::Add(
        Box::new(MuExpr::Param("mu".into())),
        Box::new(MuExpr::Binary(
            ElementwiseOp::Mul,
            Box::new(MuExpr::Param("tau".into())),
            Box::new(MuExpr::Gather {
                param_name: "z".into(),
                data_key: "group".into(),
            }),
        )),
    )
}

fn spec() -> ModelSpec {
    ModelSpec {
        dimensions: HashMap::new(),
        potentials: Vec::new(),
        deterministics: vec![("eta".into(), mean())],
        priors: vec![
            PriorSpec::Normal {
                name: "mu".into(),
                mu: HyperParam::Const(0.0),
                sigma: HyperParam::Const(5.0),
            },
            PriorSpec::HalfNormal {
                name: "tau".into(),
                sigma: HyperParam::Const(1.0),
            },
            PriorSpec::VectorNormal {
                name: "z".into(),
                n: GROUPS,
                mu: 0.0,
                sigma: 1.0,
            },
            PriorSpec::HalfNormal {
                name: "sigma".into(),
                sigma: HyperParam::Const(1.0),
            },
        ],
        likelihoods: vec![LikelihoodSpec {
            family: LikelihoodFamily::Normal,
            name: "obs".into(),
            mu_expr: mean(),
            sigma: Some(SigmaSpec::Param("sigma".into())),
            observed_key: "y".into(),
        }],
        bound_data_1d: HashMap::new(),
        bound_data_2d: HashMap::new(),
    }
}

/// Rows in groups 0 to 4 out of order; group 5 has none.
fn data(rows: usize, observed: bool) -> HashMap<String, Vec<f64>> {
    let group: Vec<f64> = (0..rows).map(|i| ((i * 7 + 3) % 5) as f64).collect();
    let mut data = HashMap::from([("group".to_string(), group.clone())]);
    if observed {
        let y = group
            .iter()
            .enumerate()
            .map(|(i, g)| 0.5 + 0.6 * g - 1.0 + ((i * 13) % 17) as f64 / 10.0)
            .collect();
        data.insert("y".to_string(), y);
    }
    data
}

fn inputs(data: &HashMap<String, Vec<f64>>) -> DataInputs {
    let mut inputs = DataInputs::default();
    for (key, values) in data {
        inputs
            .vectors
            .insert(key.clone(), Arc::from(values.as_slice()));
    }
    inputs
}

/// Every reader's output, flattened, in a fixed order.
fn readers(model: &GraphModel, fit: &ModelFit) -> Vec<(String, Vec<f64>)> {
    let graph = fit.graph();
    let mut out = Vec::new();
    for expected in [false, true] {
        let draws = fit
            .posterior_predictive(&graph, None, 11, expected)
            .unwrap();
        out.push((
            format!("posterior predictive {expected}"),
            draws.values.concat(),
        ));
        let thinned = fit
            .posterior_predictive(&graph, Some(7), 12, expected)
            .unwrap();
        out.push((
            format!("thinned predictive {expected}"),
            thinned.values.concat(),
        ));
        let new_rows = fit
            .predict(inputs(&data(9, false)), HashMap::new(), 13, expected)
            .unwrap();
        let flat = new_rows["obs"]
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        out.push((format!("predict {expected}"), flat));
    }
    let (grid, _) = fit.posterior_predictive_grid(Some(5), 14).unwrap();
    out.push(("predictive grid".into(), grid.values.concat()));
    out.push((
        "log-likelihood".into(),
        fit.log_likelihood().unwrap().concat(),
    ));
    for (label, graph) in [
        ("training", graph.clone()),
        (
            "new rows",
            fit.prediction_graph(Some(inputs(&data(9, false))), None)
                .unwrap(),
        ),
    ] {
        let values = fit.deterministics(&graph).unwrap();
        out.push((format!("deterministics {label}"), values[0].values.clone()));
    }
    let binding = model.bind(inputs(&data(40, true)), "prior").unwrap();
    let prior = model
        .prior_predictive(&binding, 25, &mut ChaCha8Rng::seed_from_u64(15))
        .unwrap();
    out.push(("prior predictive".into(), prior.predictions.concat()));
    out.push(("prior deterministics".into(), prior.deterministics.concat()));
    out
}

#[test]
fn every_reader_matches_the_unfused_path_to_the_bit() {
    let spec = spec();
    let training = data(40, true);
    let model = compile(&spec, &training, &HashMap::new())
        .expect("the model compiles")
        .into_model(&spec);
    let binding = model.bind(inputs(&training), "fit").unwrap();
    let config = SamplerConfig {
        num_chains: 2,
        num_draws: 30,
        num_warmup: 60,
        seed: 3,
        show_progress: false,
        ..Default::default()
    };
    let fit = model.sample(binding.clone(), config, None).unwrap();

    assert!(set_grouped_normal_fusion(true));
    let before = fused_grouped_normal_passes();
    let fused = readers(&model, &fit);
    let during = fused_grouped_normal_passes();
    set_grouped_normal_fusion(false);
    let separate = readers(&model, &fit);
    let after = fused_grouped_normal_passes();
    set_grouped_normal_fusion(true);
    // The fused readers ran fused forward passes, at least one per draw for
    // the log-likelihood alone; the unfused ones ran none.
    assert!(during.0 - before.0 >= 60, "{before:?} {during:?}");
    assert_eq!(after.0, during.0);

    for ((label, a), (_, b)) in fused.iter().zip(&separate) {
        assert_eq!(a.len(), b.len(), "{label}");
        assert!(!a.is_empty(), "{label}");
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: value {k}: {x} against {y}"
            );
        }
    }

    // By hand: the deterministic is `mu + tau * z[group]` at each draw, and
    // the log-likelihood the Normal density of each row about it.
    let samples = &fit.samples;
    let name = |n: &str| samples.param_names.iter().position(|p| p == n).unwrap();
    let (mu, tau, sigma) = (name("mu"), name("tau"), name("sigma"));
    let z: Vec<usize> = (0..GROUPS).map(|g| name(&format!("z[{g}]"))).collect();
    let eta = &fused
        .iter()
        .find(|(label, _)| label == "deterministics training")
        .unwrap()
        .1;
    let log_likelihood = &fused
        .iter()
        .find(|(label, _)| label == "log-likelihood")
        .unwrap()
        .1;
    let rows = training["y"].len();
    for chain in 0..2 {
        for draw in 0..30 {
            let p = &samples.samples[chain][draw];
            for i in 0..rows {
                let flat = (chain * 30 + draw) * rows + i;
                let g = training["group"][i] as usize;
                let expected = p[mu] + p[tau] * p[z[g]];
                assert!(
                    (eta[flat] - expected).abs() <= 4.0 * f64::EPSILON * expected.abs().max(1.0),
                    "deterministic {flat}: {} against {expected}",
                    eta[flat]
                );
                let d = (training["y"][i] - expected) / p[sigma];
                let density = -0.5 * std::f64::consts::TAU.ln() - p[sigma].ln() - 0.5 * d * d;
                assert!(
                    (log_likelihood[flat] - density).abs() <= 1e-12 * density.abs().max(1.0),
                    "log-likelihood {flat}: {} against {density}",
                    log_likelihood[flat]
                );
            }
        }
    }

    // The log density and gradient, which do differ in their last bits.
    let position: Vec<f64> = (0..fit.position(0, 0).len())
        .map(|k| 0.1 * k as f64 - 0.3)
        .collect();
    let fused_density = model.log_density(&binding, &position).unwrap();
    set_grouped_normal_fusion(false);
    let separate_density = model.log_density(&binding, &position).unwrap();
    set_grouped_normal_fusion(true);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
    assert!(close(fused_density.0, separate_density.0));
    for (a, b) in fused_density.1.iter().zip(&separate_density.1) {
        assert!(close(*a, *b), "{a} against {b}");
    }
}
