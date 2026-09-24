//! Model batches keep every cell's outcome: a cell whose reported draws cannot
//! be derived is that cell's error, not the batch's, when errors are
//! collected.
use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::{
    sample_model_batch, DisplayParamSpec, GraphModel, HyperParam, ModelBatchCell,
};
use rustmc_core::sampler::{BatchSampleConfig, BatchSeedPolicy, BoundBatchOptions};
use std::collections::HashMap;
use std::sync::Arc;

const ARTIFACT: &str = include_str!("fixtures/graph_model_v1.json");

fn inputs(rows: usize) -> DataInputs {
    let mut data = DataInputs::default();
    let design = (0..rows)
        .flat_map(|i| [1., i as f64 / 3.])
        .collect::<Vec<_>>();
    data.matrices.insert(
        "X".into(),
        MatrixBinding {
            data: Arc::from(design),
            n_rows: rows,
            n_cols: 2,
        },
    );
    let y = (0..rows).map(|i| i as f64 / 2.).collect::<Vec<_>>();
    data.vectors.insert("y".into(), Arc::from(y));
    data
}

fn config() -> BatchSampleConfig {
    BatchSampleConfig {
        num_draws: 10,
        num_warmup: 10,
        show_progress: false,
        ..Default::default()
    }
}

fn options(collect_errors: bool) -> BoundBatchOptions {
    BoundBatchOptions {
        threads: 2,
        chunk_size: 8,
        collect_errors,
        seed_policy: BatchSeedPolicy::CellIdV1,
    }
}

/// A model whose sampling succeeds but whose reported parameters cannot be
/// derived: its display layer names a hyperparameter the model lacks.
fn undisplayable(model: &GraphModel) -> GraphModel {
    let mut broken = model.clone();
    broken.display_params[0] = DisplayParamSpec::DerivedNonCenteredNormal {
        name: "alpha".into(),
        raw_index: 0,
        mu: HyperParam::Param("missing".into()),
        sigma: HyperParam::Const(1.0),
    };
    broken
}

fn cells(model: &GraphModel) -> Vec<ModelBatchCell> {
    let good = |id: &str| ModelBatchCell {
        id: id.into(),
        model: model.clone(),
        binding: model.bind(inputs(6), id).map_err(|e| e.to_string()),
        initial: None,
    };
    vec![
        good("a"),
        ModelBatchCell {
            model: undisplayable(model),
            ..good("b")
        },
        ModelBatchCell {
            binding: Err("could not bind".into()),
            ..good("c")
        },
        good("d"),
    ]
}

#[test]
fn collected_batches_keep_a_cell_whose_draws_cannot_be_reported() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let results = sample_model_batch(cells(&model), config(), options(true)).unwrap();
    assert_eq!(results.len(), 4);
    assert!(results[0].is_ok() && results[3].is_ok());
    assert!(
        results[1].as_ref().unwrap_err().contains("missing"),
        "{:?}",
        results[1].as_ref().err()
    );
    assert_eq!(results[2].as_ref().unwrap_err(), "could not bind");
}

#[test]
fn raising_batches_name_the_first_failed_cell() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let mut cells = cells(&model);
    cells.remove(2);
    let error = sample_model_batch(cells, config(), options(false)).unwrap_err();
    assert!(error.to_string().starts_with("dataset 'b': "), "{error}");
}

#[test]
fn graph_model_batches_match_their_single_fits() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let bindings = ["x", "y"]
        .into_iter()
        .map(|id| {
            (
                id.to_string(),
                model.bind(inputs(5), id).map_err(|e| e.to_string()),
            )
        })
        .collect();
    let options = BoundBatchOptions {
        seed_policy: BatchSeedPolicy::PositionV0,
        ..options(true)
    };
    let batch = model
        .sample_batch(bindings, config(), options, HashMap::new())
        .unwrap();
    for (index, fit) in batch.iter().enumerate() {
        let single = model
            .sample(
                model.bind(inputs(5), "single").unwrap(),
                rustmc_core::sampler::SamplerConfig {
                    num_chains: 1,
                    num_draws: 10,
                    num_warmup: 10,
                    max_tree_depth: 8,
                    seed: 42u64.wrapping_add((index as u64) << 32),
                    show_progress: false,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        assert_eq!(
            fit.as_ref().unwrap().samples.samples,
            single.samples.samples
        );
    }
}
