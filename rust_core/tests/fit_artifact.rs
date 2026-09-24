//! `rustmc.graph-fit` artifacts written by the Python bindings load, evaluate
//! and re-save from Rust. The fixtures under the repository's `tests/fixtures`
//! are shared with the Python suite (`tests/test_fit_artifact.py`), which
//! asserts the same bytes, so a format change on either side fails both.
use std::collections::HashMap;

use rustmc_core::data::DataInputs;
use rustmc_core::model::ModelFit;

const UNIFORM: &str = include_str!("../../tests/fixtures/graph_fit_v2_uniform.json");
const REGRESSION: &str = include_str!("../../tests/fixtures/graph_fit_v2_regression.json");

#[test]
fn loads_the_python_written_uniform_fixture() {
    let fit = ModelFit::from_json(UNIFORM).unwrap();
    assert_eq!(fit.samples.param_names, ["u"]);
    assert_eq!((fit.num_chains(), fit.num_draws()), (1, 3));
    // The stored exact positions, not the inverse of the rounded draws.
    let positions = fit.raw_samples().unconstrained_samples.as_ref().unwrap();
    assert_eq!(positions[0][2], [-0.4889066992204638]);
    let reloaded = ModelFit::from_json(&fit.to_json().unwrap()).unwrap();
    assert_eq!(reloaded.samples.samples, fit.samples.samples);
}

#[test]
fn python_and_rust_write_the_same_bytes() {
    let fit = ModelFit::from_json(REGRESSION).unwrap();
    assert_eq!(fit.to_json().unwrap(), REGRESSION);
}

#[test]
fn restored_fit_scores_its_training_data() {
    let fit = ModelFit::from_json(REGRESSION).unwrap();
    let (vectors, _) = fit.training_data().unwrap();
    let (x, y) = (&vectors["x"], &vectors["y"]);
    let log_likelihood = fit.log_likelihood().unwrap();
    assert_eq!(log_likelihood.len(), 1);
    let n_obs = y.len();
    assert_eq!(
        log_likelihood[0].len(),
        fit.num_chains() * fit.num_draws() * n_obs
    );
    let names = &fit.samples.param_names;
    let index = |name: &str| names.iter().position(|n| n == name).unwrap();
    for (row, draw) in fit.samples.samples.iter().flatten().enumerate() {
        let (a, beta, s) = (draw[index("a")], draw[index("beta")], draw[index("s")]);
        for i in 0..n_obs {
            let z = (y[i] - a - beta * x[i]) / s;
            let expected = -0.5 * z * z - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
            let actual = log_likelihood[0][row * n_obs + i];
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "draw {row}, obs {i}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn restored_fit_predicts_at_new_inputs() {
    let fit = ModelFit::from_json(REGRESSION).unwrap();
    let inputs = DataInputs {
        vectors: [("x".to_string(), vec![0.0, 2.0].into())].into(),
        matrices: HashMap::new(),
    };
    let expected = fit
        .predict(inputs.clone(), HashMap::new(), 3, true)
        .unwrap();
    let names = &fit.samples.param_names;
    let (a, beta) = (
        names.iter().position(|n| n == "a").unwrap(),
        names.iter().position(|n| n == "beta").unwrap(),
    );
    for (chain, draws) in expected["y_obs"].iter().enumerate() {
        for (draw, means) in draws.iter().enumerate() {
            let params = &fit.samples.samples[chain][draw];
            assert_eq!(means, &[params[a], params[a] + 2.0 * params[beta]]);
        }
    }
    // Seeded simulation reproduces.
    let first = fit
        .predict(inputs.clone(), HashMap::new(), 3, false)
        .unwrap();
    let second = fit.predict(inputs, HashMap::new(), 3, false).unwrap();
    assert_eq!(first, second);
}

#[test]
fn corrupted_positions_are_refused() {
    let mut artifact: serde_json::Value = serde_json::from_str(REGRESSION).unwrap();
    artifact["posterior"]["unconstrained_samples"][0][0][2] = serde_json::json!(5.0);
    let error = ModelFit::from_json(&artifact.to_string()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unconstrained posterior positions disagree"),
        "{error}"
    );
}
