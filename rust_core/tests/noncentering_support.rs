use rustmc_core::data::DataInputs;
use rustmc_core::model::{GraphModel, HyperParam, ModelArtifact, ModelSpec, PriorSpec};
use std::collections::HashMap;

#[test]
fn noncentering_preserves_referenced_scale_support_through_artifacts() {
    let definition = ModelSpec {
        dimensions: HashMap::new(),
        potentials: vec![],
        deterministics: vec![],
        priors: vec![
            PriorSpec::Normal {
                name: "scale".into(),
                mu: HyperParam::Const(0.0),
                sigma: HyperParam::Const(1.0),
            },
            PriorSpec::Normal {
                name: "effect".into(),
                mu: HyperParam::Const(1e20),
                sigma: HyperParam::Param("scale".into()),
            },
        ],
        likelihoods: vec![],
        bound_data_1d: HashMap::new(),
        bound_data_2d: HashMap::new(),
    };
    let model = GraphModel::from_artifact(ModelArtifact {
        format: "rustmc.graph-model".into(),
        version: 1,
        definition,
        schema: Default::default(),
    })
    .unwrap();
    let restored = GraphModel::from_json(&model.to_json().unwrap()).unwrap();
    for model in [model, restored] {
        let data = model.bind(DataInputs::default(), "empty").unwrap();
        for scale in [-1.0, 0.0] {
            assert_eq!(
                model.log_density(&data, &[scale, 0.3]).unwrap().0,
                f64::NEG_INFINITY
            );
        }
        for scale in [1e-12, 0.5, 1.0, 10.0] {
            let (lp, grad) = model.log_density(&data, &[scale, 0.3]).unwrap();
            let expected = -(2.0 * std::f64::consts::PI).ln() - 0.5 * (scale * scale + 0.09);
            assert!((lp - expected).abs() < 1e-12);
            assert_eq!(grad, vec![-scale, -0.3]);
        }
    }
}
