use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::GraphModel;
use rustmc_core::sampler::SamplerConfig;
use std::collections::HashMap;
use std::sync::Arc;
const ARTIFACT: &str = include_str!("fixtures/graph_model_v1.json");
fn inputs(rows: usize, observed: bool) -> DataInputs {
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
    if observed {
        data.vectors.insert(
            "y".into(),
            Arc::from((0..rows).map(|i| i as f64 / 2.).collect::<Vec<_>>()),
        );
    }
    data
}
#[test]
fn old_python_artifact_roundtrips_and_gradients_match_finite_differences() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let decoded = GraphModel::from_json(&model.to_json().unwrap()).unwrap();
    assert!(decoded.definition.bound_data_1d.is_empty());
    assert!(decoded.definition.bound_data_2d.is_empty());
    for rows in [2, 7] {
        let binding = model.bind(inputs(rows, true), "fit").unwrap();
        let q = vec![0.2, -0.3, 0.4, -0.2];
        let (lp, grad) = model.log_density(&binding, &q).unwrap();
        assert_eq!(
            (lp, grad.clone()),
            decoded.log_density(&binding, &q).unwrap()
        );
        for i in 0..q.len() {
            let mut plus = q.clone();
            plus[i] += 1e-6;
            let mut minus = q.clone();
            minus[i] -= 1e-6;
            let difference = (model.log_density(&binding, &plus).unwrap().0
                - model.log_density(&binding, &minus).unwrap().0)
                / 2e-6;
            assert!((grad[i] - difference).abs() < 1e-5);
        }
    }
}
#[test]
fn native_fit_predicts_new_rows_and_validates_bindings() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let data = model.bind(inputs(7, true), "fit").unwrap();
    let fit = model
        .sample(
            data,
            SamplerConfig {
                num_chains: 2,
                num_draws: 40,
                num_warmup: 100,
                show_progress: false,
                ..Default::default()
            },
            None,
        )
        .unwrap();
    let draws = fit
        .predict(inputs(3, false), HashMap::new(), 43, false)
        .unwrap();
    assert_eq!(draws["obs"].len(), 2);
    assert_eq!(draws["obs"][0].len(), 40);
    assert_eq!(draws["obs"][0][0].len(), 3);
    assert_eq!(
        draws,
        fit.predict(inputs(3, false), HashMap::new(), 43, false)
            .unwrap()
    );
    assert!(model.bind(inputs(3, false), "bad").is_err());
    assert!(fit
        .predict(DataInputs::default(), HashMap::new(), 43, false)
        .is_err());
}
#[test]
fn corrupt_artifact_and_wrong_parameter_axes_are_rejected() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let binding = model.bind(inputs(3, true), "x").unwrap();
    assert!(model.log_density(&binding, &[0.]).is_err());
    let mut artifact: serde_json::Value = serde_json::from_str(ARTIFACT).unwrap();
    artifact["version"] = 2.into();
    assert!(GraphModel::from_json(&artifact.to_string()).is_err());
    artifact["version"] = 1.into();
    artifact["schema"]["matrices"][0]["kind"]["Matrix"]["n_cols"] = 3.into();
    assert!(GraphModel::from_json(&artifact.to_string()).is_err());
}
