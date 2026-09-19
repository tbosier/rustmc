//! A Rust-only caller must be able to simulate from the prior of a model it
//! loaded from an artifact, without going through the Python bindings.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::{DisplayParamSpec, GraphModel};
use std::sync::Arc;

const ARTIFACT: &str = include_str!("fixtures/graph_model_v1.json");
const ROWS: usize = 6;

/// The shipped fixture carries a potential, which has no random generator.
/// Drop it to get an artifact whose prior is simulable, leaving everything
/// else - including the schema the loader cross-checks - untouched.
fn simulable_artifact() -> String {
    let mut artifact: serde_json::Value = serde_json::from_str(ARTIFACT).unwrap();
    assert!(!artifact["definition"]["potentials"]
        .as_array()
        .unwrap()
        .is_empty());
    artifact["definition"]["potentials"] = serde_json::json!([]);
    artifact.to_string()
}

fn inputs() -> DataInputs {
    let mut data = DataInputs::default();
    data.matrices.insert(
        "X".into(),
        MatrixBinding {
            data: Arc::from(
                (0..ROWS)
                    .flat_map(|i| [1., i as f64 / 3.])
                    .collect::<Vec<_>>(),
            ),
            n_rows: ROWS,
            n_cols: 2,
        },
    );
    data.vectors.insert(
        "y".into(),
        Arc::from((0..ROWS).map(|i| i as f64 / 2.).collect::<Vec<_>>()),
    );
    data
}

fn display_names(model: &GraphModel) -> Vec<String> {
    model
        .display_params
        .iter()
        .map(|spec| match spec {
            DisplayParamSpec::Raw { name, .. }
            | DisplayParamSpec::DerivedNonCenteredNormal { name, .. } => name.clone(),
        })
        .collect()
}

#[test]
fn loaded_artifact_simulates_its_prior_predictive_from_rust() {
    let model = GraphModel::from_json(&simulable_artifact()).unwrap();
    let binding = model.bind(inputs(), "prior").unwrap();
    let n_samples = 2000;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let draws = model
        .prior_predictive(&binding, n_samples, &mut rng)
        .unwrap();

    // alpha, beta[0], beta[1], sigma
    let names = display_names(&model);
    assert_eq!(names, ["alpha", "beta[0]", "beta[1]", "sigma"]);
    assert_eq!(draws.params.len(), names.len());
    assert!(draws.params.iter().all(|col| col.len() == n_samples));

    // One likelihood, one row of predictions per observation per draw.
    assert_eq!(model.likelihood_names, ["obs"]);
    assert_eq!(draws.n_obs, [ROWS]);
    assert_eq!(draws.predictions.len(), 1);
    assert_eq!(draws.predictions[0].len(), n_samples * ROWS);
    assert!(draws.predictions[0].iter().all(|y| y.is_finite()));

    // The `signal` deterministic is a vector of length ROWS.
    assert_eq!(draws.deterministic_lens, [ROWS]);
    assert_eq!(draws.deterministics[0].len(), n_samples * ROWS);

    // The draws must come from the declared priors, not from some default.
    let mean = |col: &[f64]| col.iter().sum::<f64>() / col.len() as f64;
    let sd = |col: &[f64]| {
        let m = mean(col);
        (col.iter().map(|x| (x - m).powi(2)).sum::<f64>() / col.len() as f64).sqrt()
    };
    // alpha ~ Normal(0, 2)
    assert!(mean(&draws.params[0]).abs() < 0.2, "alpha mean drifted");
    assert!((sd(&draws.params[0]) - 2.0).abs() < 0.2, "alpha sd drifted");
    // beta ~ Normal(0, 1), two independent components
    for col in &draws.params[1..3] {
        assert!(mean(col).abs() < 0.1);
        assert!((sd(col) - 1.0).abs() < 0.1);
    }
    // sigma ~ HalfNormal(1): strictly positive, mean sqrt(2/pi)
    assert!(draws.params[3].iter().all(|&s| s > 0.0));
    assert!((mean(&draws.params[3]) - (2.0 / std::f64::consts::PI).sqrt()).abs() < 0.06);

    // Same seed, same draws.
    let mut again = ChaCha8Rng::seed_from_u64(7);
    assert_eq!(
        draws,
        model
            .prior_predictive(&binding, n_samples, &mut again)
            .unwrap()
    );
    // A different seed must not give the same numbers.
    let mut other = ChaCha8Rng::seed_from_u64(8);
    assert_ne!(
        draws,
        model
            .prior_predictive(&binding, n_samples, &mut other)
            .unwrap()
    );
}

#[test]
fn single_prior_draw_needs_no_data_and_leads_the_predictive_stream() {
    let model = GraphModel::from_json(&simulable_artifact()).unwrap();

    // `sample_prior` works off the artifact alone - no binding, no fit.
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let draw = model.sample_prior(&mut rng).unwrap();
    assert_eq!(draw.raw.len(), model.structure.param_count);
    assert_eq!(draw.display.len(), model.display_params.len());
    assert!(draw.raw.iter().chain(&draw.display).all(|x| x.is_finite()));
    // `sigma` is the fourth display parameter and is exp-transformed, so the
    // raw value is a log scale and the display value its positive exponent.
    assert!(draw.display[3] > 0.0);
    assert!((draw.display[3].ln() - draw.raw[3]).abs() < 1e-12);

    // The predictive stream starts from exactly that draw.
    let binding = model.bind(inputs(), "prior").unwrap();
    let mut stream = ChaCha8Rng::seed_from_u64(11);
    let predictive = model.prior_predictive(&binding, 3, &mut stream).unwrap();
    for (pi, &value) in draw.display.iter().enumerate() {
        assert_eq!(predictive.params[pi][0], value);
    }
}

#[test]
fn a_model_with_a_potential_has_no_prior_to_simulate() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let binding = model.bind(inputs(), "prior").unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let error = model
        .prior_predictive(&binding, 1, &mut rng)
        .expect_err("a potential has no random generator");
    assert!(
        error.to_string().contains("potentials"),
        "unexpected message: {error}"
    );

    // A potential is part of the target, so the priors alone are not the
    // prior. The single-draw entry point must refuse for the same reason.
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let error = model
        .sample_prior(&mut rng)
        .expect_err("a potential has no random generator");
    assert!(
        error.to_string().contains("potentials"),
        "unexpected message: {error}"
    );
}

/// Load a model the way a user would: compile a definition, publish it as an
/// artifact, and read that artifact back.
fn model_from_definition(definition: serde_json::Value, n_cols: usize) -> GraphModel {
    use rustmc_core::model::{compile, ModelArtifact, ModelSpec};
    use std::collections::HashMap;
    let definition: ModelSpec = serde_json::from_value(definition).unwrap();
    let vectors = HashMap::from([("y".to_string(), vec![0.0; ROWS])]);
    let matrices = if n_cols == 0 {
        HashMap::new()
    } else {
        HashMap::from([("X".to_string(), (vec![0.0; ROWS * n_cols], ROWS, n_cols))])
    };
    let compiled = compile(&definition, &vectors, &matrices).unwrap();
    let artifact = ModelArtifact {
        format: "rustmc.graph-model".into(),
        version: 1,
        definition,
        schema: compiled.graph.schema.clone(),
    };
    GraphModel::from_json(&serde_json::to_string(&artifact).unwrap()).unwrap()
}

fn matrix_inputs(n_cols: usize) -> DataInputs {
    let mut data = DataInputs::default();
    data.matrices.insert(
        "X".into(),
        MatrixBinding {
            data: Arc::from(
                (0..ROWS * n_cols)
                    .map(|i| (i % 5) as f64 - 2.0)
                    .collect::<Vec<_>>(),
            ),
            n_rows: ROWS,
            n_cols,
        },
    );
    data.vectors.insert(
        "y".into(),
        Arc::from((0..ROWS).map(|i| i as f64 / 2.).collect::<Vec<_>>()),
    );
    data
}

#[test]
fn a_matrix_backed_parameter_is_auto_promoted_from_the_schema_alone() {
    // `beta` is declared as one scalar Normal but used as `X @ beta`, so the
    // model really has one parameter per column of `X`. Nothing in the
    // definition says three; only the stored schema's column count does. A
    // loader that failed to recover that would draw a single value, which
    // would not fit the parameter axis.
    const COLS: usize = 3;
    let model = model_from_definition(
        serde_json::json!({
            "dimensions": {}, "potentials": [], "deterministics": [],
            "priors": [
                {"Normal": {"name": "beta", "mu": {"Const": 0.0}, "sigma": {"Const": 1.0}}},
                {"HalfNormal": {"name": "scale", "sigma": {"Const": 1.0}}}
            ],
            "likelihoods": [{
                "family": "Normal", "name": "obs",
                "mu_expr": {"MatVec": {"param_name": "beta", "data_key": "X"}},
                "sigma": {"Param": "scale"}, "observed_key": "y"
            }]
        }),
        COLS,
    );
    assert_eq!(
        display_names(&model),
        ["beta[0]", "beta[1]", "beta[2]", "scale"]
    );
    assert_eq!(model.structure.param_count, COLS + 1);

    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let draw = model.sample_prior(&mut rng).unwrap();
    assert_eq!(draw.raw.len(), COLS + 1);
    // Independent standard Normals, so the components must not coincide.
    assert!(draw.raw[0] != draw.raw[1] && draw.raw[1] != draw.raw[2]);

    let binding = model.bind(matrix_inputs(COLS), "prior").unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let draws = model.prior_predictive(&binding, 500, &mut rng).unwrap();
    assert_eq!(draws.params.len(), COLS + 1);
    for column in &draws.params[..COLS] {
        let mean = column.iter().sum::<f64>() / column.len() as f64;
        let var = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
        assert!(mean.abs() < 0.2, "beta component mean drifted: {mean}");
        assert!((var.sqrt() - 1.0).abs() < 0.2, "beta component sd drifted");
    }
}

#[test]
fn a_hierarchical_normal_is_drawn_in_its_noncentred_coordinates() {
    // `theta ~ Normal(mu, tau)` with parameter hyperparameters is stored
    // noncentred: the raw value is the standard Normal z and the reported
    // value is mu + tau * z. The draw must report both consistently, or the
    // prior predictive and the posterior disagree about what `theta` means.
    let model = model_from_definition(
        serde_json::json!({
            "dimensions": {}, "potentials": [], "deterministics": [],
            "priors": [
                {"Normal": {"name": "mu", "mu": {"Const": 0.0}, "sigma": {"Const": 1.0}}},
                {"HalfNormal": {"name": "tau", "sigma": {"Const": 1.0}}},
                {"Normal": {"name": "theta", "mu": {"Param": "mu"}, "sigma": {"Param": "tau"}}}
            ],
            "likelihoods": [{
                "family": "Normal", "name": "obs", "mu_expr": {"Param": "theta"},
                "sigma": {"Const": 1.0}, "observed_key": "y"
            }]
        }),
        0,
    );
    assert_eq!(display_names(&model), ["mu", "tau", "theta"]);
    let raw_index = match &model.display_params[2] {
        DisplayParamSpec::DerivedNonCenteredNormal { raw_index, .. } => *raw_index,
        other => panic!("theta was not stored noncentred: {other:?}"),
    };

    for seed in [0, 1, 99] {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let draw = model.sample_prior(&mut rng).unwrap();
        let (mu, tau, theta) = (draw.display[0], draw.display[1], draw.display[2]);
        assert!(tau > 0.0);
        assert_eq!(theta, mu + tau * draw.raw[raw_index]);
        // The stored coordinate is the standard Normal z, not theta itself.
        assert_ne!(draw.raw[raw_index], theta);
    }
}

/// Exact draws pinned to this RNG stream. Reordering, adding or dropping a
/// single RNG call moves every number after it, so this fails on any change
/// that would quietly make a seed mean different draws - which the earlier
/// assertions, comparing the implementation against itself, cannot catch.
///
/// The values were taken from the first build after prior generation moved
/// into the core; that move was separately checked to leave the Python
/// `sample_prior_predictive` output byte-for-byte unchanged.
#[test]
fn the_draw_stream_for_a_given_seed_is_pinned() {
    let model = GraphModel::from_json(&simulable_artifact()).unwrap();
    let binding = model.bind(inputs(), "prior").unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let draws = model.prior_predictive(&binding, 2, &mut rng).unwrap();
    let flat: Vec<f64> = draws
        .params
        .iter()
        .flatten()
        .chain(draws.predictions.iter().flatten())
        .chain(draws.deterministics.iter().flatten())
        .copied()
        .collect();
    if std::env::var_os("RUSTMC_PRINT_GOLDEN").is_some() {
        for value in &flat {
            println!("{value:?},");
        }
    }
    assert_eq!(flat.len(), GOLDEN.len());
    for (i, (got, want)) in flat.iter().zip(GOLDEN).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "draw {i}: {got} != {want}");
    }
}

#[rustfmt::skip]
const GOLDEN: &[f64] = &[
    // 4 display parameters x 2 draws
    -1.5507438664355941, -2.041557941343919,
    -1.3834217200084091, -0.25541460655939807,
    0.8897130187430372, 0.10674006979725632,
    0.3597790583440233, 0.34009124420128517,
    // `obs`, 2 draws x 6 rows
    -2.826228629705682, -2.8714460949157363, -2.74731464099498,
    -1.6551544848624244, -1.8449351847677886, -0.8883108803052228,
    -2.342822447087418, -2.2222825857650075, -2.494810438764695,
    -2.579069971897193, -2.3088476868521735, -2.387144911790916,
    // `signal`, 2 draws x 6 rows
    -2.9341655864440033, -2.6375945801963243, -2.341023573948645,
    -2.044452567700966, -1.747881561453287, -1.4513105552056078,
    -2.2969725479033167, -2.261392524637565, -2.225812501371813,
    -2.1902324781060605, -2.1546524548403085, -2.1190724315745566,
];
