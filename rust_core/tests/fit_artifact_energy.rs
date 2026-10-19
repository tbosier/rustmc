//! Version 3 of the `rustmc.graph-fit` artifact keeps the energy telemetry
//! E-BFMI and tree-depth saturation need, and version 4 also why each NUTS
//! tree stopped. Fits from older artifacts, which lack some of it, still
//! load, report what they lack as unknown, and save as the newest version
//! that holds what they have, which older readers load.
use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::{GraphModel, ModelFit};
use rustmc_core::sampler::{SamplerConfig, SamplerType};
use serde_json::Value;
use std::sync::Arc;

const MODEL: &str = include_str!("fixtures/graph_model_v1.json");

fn fit(sampler: SamplerType) -> ModelFit {
    let model = GraphModel::from_json(MODEL).unwrap();
    let rows = 6;
    let mut data = DataInputs::default();
    data.matrices.insert(
        "X".into(),
        MatrixBinding {
            data: Arc::from(
                (0..rows)
                    .flat_map(|i| [1., i as f64 / 3.])
                    .collect::<Vec<_>>(),
            ),
            n_rows: rows,
            n_cols: 2,
        },
    );
    data.vectors.insert(
        "y".into(),
        Arc::from((0..rows).map(|i| i as f64 / 2.).collect::<Vec<_>>()),
    );
    let binding = model.bind(data, "fit").unwrap();
    model
        .sample(
            binding,
            SamplerConfig {
                sampler,
                num_chains: 2,
                num_draws: 30,
                num_warmup: 40,
                max_tree_depth: 7,
                show_progress: false,
                seed: 9,
                ..Default::default()
            },
            None,
        )
        .unwrap()
}

fn json(fit: &ModelFit) -> Value {
    serde_json::from_str(&fit.to_json().unwrap()).unwrap()
}

fn load(artifact: &Value) -> Result<ModelFit, String> {
    ModelFit::from_json(&artifact.to_string()).map_err(|error| error.to_string())
}

fn energies(fit: &ModelFit) -> Vec<u64> {
    fit.raw_samples()
        .transitions
        .iter()
        .flatten()
        .map(|t| t.energy.to_bits())
        .collect()
}

#[test]
fn version_four_round_trips_energy_terminations_and_the_tree_depth_limit() {
    for (sampler, limit) in [(SamplerType::Nuts, Some(7)), (SamplerType::Hmc, None)] {
        let original = fit(sampler);
        assert!(original
            .raw_samples()
            .transitions
            .iter()
            .flatten()
            .all(|t| t.energy.is_finite()));
        let artifact = json(&original);
        assert_eq!(artifact["version"], 4);
        let reasons: Vec<&Value> = artifact["posterior"]["transitions"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|chain| chain.as_array().unwrap())
            .map(|t| &t["termination"])
            .collect();
        match sampler {
            SamplerType::Nuts => assert!(reasons.iter().all(|r| r.is_string())),
            _ => assert!(reasons.iter().all(|r| r.is_null())),
        }
        assert_eq!(
            artifact["posterior"]["max_tree_depth"].as_u64(),
            limit.map(|l| l as u64)
        );
        let restored = load(&artifact).unwrap();
        assert_eq!(energies(&restored), energies(&original));
        assert_eq!(restored.raw_samples().max_tree_depth, limit);
        let (before, after) = (
            original.raw_samples().transition_diagnostics(),
            restored.raw_samples().transition_diagnostics(),
        );
        for (a, b) in before.chains.iter().zip(&after.chains) {
            assert_eq!(a.e_bfmi.to_bits(), b.e_bfmi.to_bits());
            assert_eq!(a.tree_depth_saturation, b.tree_depth_saturation);
            assert_eq!(a.max_tree_depth_hits, b.max_tree_depth_hits);
            assert_eq!(
                a.max_tree_depth_hits.is_some(),
                sampler == SamplerType::Nuts
            );
        }
        assert_eq!(json(&restored), artifact);
    }
}

fn strip_terminations(artifact: &mut Value) {
    for chain in artifact["posterior"]["transitions"].as_array_mut().unwrap() {
        for transition in chain.as_array_mut().unwrap() {
            transition.as_object_mut().unwrap().remove("termination");
        }
    }
}

#[test]
fn version_three_loads_without_terminations_and_saves_as_version_three() {
    let original = fit(SamplerType::Nuts);
    let mut artifact = json(&original);
    artifact["version"] = 3.into();
    strip_terminations(&mut artifact);
    let restored = load(&artifact).unwrap();
    assert!(restored
        .raw_samples()
        .transitions
        .iter()
        .flatten()
        .all(|t| t.termination.is_none()));
    let report = restored.raw_samples().transition_diagnostics();
    assert_eq!(report.total_max_tree_depth_hits, None);
    assert_eq!(
        report.total_tree_depth_saturation,
        original
            .raw_samples()
            .transition_diagnostics()
            .total_tree_depth_saturation
    );
    // Version 3 holds everything this fit has, so readers of it still load it.
    assert_eq!(json(&restored), artifact);
}

#[test]
fn older_versions_load_without_energy_and_save_as_version_two() {
    let mut artifact = json(&fit(SamplerType::Nuts));
    artifact["version"] = 2.into();
    artifact["posterior"]
        .as_object_mut()
        .unwrap()
        .remove("max_tree_depth");
    for chain in artifact["posterior"]["transitions"].as_array_mut().unwrap() {
        for transition in chain.as_array_mut().unwrap() {
            transition.as_object_mut().unwrap().remove("energy");
        }
    }
    strip_terminations(&mut artifact);
    let restored = load(&artifact).unwrap();
    assert!(energies(&restored)
        .iter()
        .all(|bits| f64::from_bits(*bits).is_nan()));
    let report = restored.raw_samples().transition_diagnostics();
    assert!(report.chains.iter().all(|chain| chain.e_bfmi.is_nan()));
    assert_eq!(report.total_tree_depth_saturation, None);
    // Nothing to store in version 3, so readers before it can still load it.
    assert_eq!(json(&restored), artifact);
}

#[test]
fn inconsistent_energy_telemetry_is_rejected() {
    let nuts = json(&fit(SamplerType::Nuts));
    let hmc = json(&fit(SamplerType::Hmc));
    type Corruption = fn(&mut Value);
    let cases: [(&str, &Value, Corruption, &str); 17] = [
        (
            "a depth-1 tree with more than one leapfrog step",
            &nuts,
            |a| {
                let transition = a["posterior"]["transitions"][0]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|t| t["num_leapfrog_steps"].as_u64().unwrap() > 1)
                    .unwrap();
                transition["tree_depth"] = 1.into();
            },
            "disagrees with its transition's leapfrog step count",
        ),
        (
            "fewer leapfrog steps than the full doublings before its last",
            &nuts,
            |a| {
                let transition = a["posterior"]["transitions"][0]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|t| t["tree_depth"].as_u64().unwrap() >= 3)
                    .unwrap();
                let depth = transition["tree_depth"].as_u64().unwrap();
                transition["num_leapfrog_steps"] = ((1u64 << (depth - 1)) - 1).into();
            },
            "disagrees with its transition's leapfrog step count",
        ),
        (
            "more leapfrog steps than a tree of its depth holds",
            &nuts,
            |a| {
                let transition = &mut a["posterior"]["transitions"][1][5];
                let depth = transition["tree_depth"].as_u64().unwrap();
                transition["num_leapfrog_steps"] = (1u64 << depth).into();
            },
            "disagrees with its transition's leapfrog step count",
        ),
        (
            "a draw energy below the potential energy at its draw",
            &hmc,
            |a| {
                let warmup = a["posterior"]["transitions"][0].as_array().unwrap().len()
                    - a["posterior"]["samples"][0].as_array().unwrap().len();
                a["posterior"]["transitions"][0][warmup + 2]["energy"] = (-1.0e6).into();
            },
            "below the potential energy",
        ),
        (
            "a tree depth of 0, which no NUTS transition has",
            &nuts,
            |a| {
                a["posterior"]["transitions"][1][4]["tree_depth"] = 0.into();
            },
            "maximum tree depth disagrees",
        ),
        (
            "version 3 without the exact positions of a transformed parameter",
            &nuts,
            |a| {
                a["posterior"]
                    .as_object_mut()
                    .unwrap()
                    .remove("unconstrained_samples");
            },
            "require unconstrained posterior positions",
        ),
        (
            "an energy missing in version 3",
            &nuts,
            |a| {
                a["posterior"]["transitions"][0][3]
                    .as_object_mut()
                    .unwrap()
                    .remove("energy");
            },
            "finite energy for every transition",
        ),
        (
            "energies in version 2",
            &nuts,
            |a| {
                a["version"] = 2.into();
                a["posterior"]
                    .as_object_mut()
                    .unwrap()
                    .remove("max_tree_depth");
            },
            "need artifact version 3",
        ),
        (
            "a NUTS fit without its tree-depth limit",
            &nuts,
            |a| {
                a["posterior"]
                    .as_object_mut()
                    .unwrap()
                    .remove("max_tree_depth");
            },
            "maximum tree depth disagrees",
        ),
        (
            "a limit below a recorded tree depth",
            &nuts,
            |a| {
                a["posterior"]["max_tree_depth"] = 1.into();
            },
            "maximum tree depth disagrees",
        ),
        (
            "a limit on an HMC fit",
            &hmc,
            |a| {
                a["posterior"]["max_tree_depth"] = 10.into();
            },
            "maximum tree depth disagrees",
        ),
        (
            "an unknown version",
            &nuts,
            |a| {
                a["version"] = 5.into();
            },
            "unsupported format/version",
        ),
        (
            "a NUTS transition without its termination reason in version 4",
            &nuts,
            |a| {
                a["posterior"]["transitions"][0][3]
                    .as_object_mut()
                    .unwrap()
                    .remove("termination");
            },
            "termination reason for every NUTS transition",
        ),
        (
            "a termination reason on an HMC transition",
            &hmc,
            |a| {
                a["posterior"]["transitions"][0][3]["termination"] = "u_turn".into();
            },
            "termination reason for every NUTS transition",
        ),
        (
            "termination reasons in version 3",
            &nuts,
            |a| {
                a["version"] = 3.into();
            },
            "need artifact version 4",
        ),
        (
            "a divergence reason on a transition that did not diverge",
            &nuts,
            |a| {
                let transition = a["posterior"]["transitions"][0]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|t| t["divergent"] == false)
                    .unwrap();
                transition["termination"] = "divergence".into();
            },
            "disagrees with its transition's tree",
        ),
        (
            "a tree that ran out of depth below the limit",
            &nuts,
            |a| {
                let transition = a["posterior"]["transitions"][1]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|t| t["tree_depth"].as_u64().unwrap() < 7 && t["divergent"] == false)
                    .unwrap();
                transition["termination"] = "max_tree_depth".into();
            },
            "disagrees with its transition's tree",
        ),
    ];
    for (label, source, corrupt, reason) in cases {
        let mut artifact = source.clone();
        corrupt(&mut artifact);
        match load(&artifact) {
            Ok(_) => panic!("{label} was accepted"),
            Err(error) => assert!(error.contains(reason), "{label}: {error}"),
        }
    }
    // The uncorrupted artifacts load, so each rejection is the corruption's;
    // a limit of 1 is below a recorded depth only if some tree went deeper.
    assert!(nuts["posterior"]["transitions"][0]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["tree_depth"].as_u64().unwrap() > 1));
    load(&nuts).unwrap();
    load(&hmc).unwrap();
}
