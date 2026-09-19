//! An artifact from a newer or corrupted writer must be rejected, not silently
//! truncated. Serde drops unknown fields by default, and the loader's
//! `compiled.graph.schema != artifact.schema` cross-check runs *after* that, so
//! anything serde discarded is invisible to it.
//!
//! The same must hold for every other artifact this crate loads: the structural
//! model, the structural fit, and the dynamic GLM fit. Each has its own
//! `serde_json::from_str` and so its own unknown-field policy to state.
use rustmc_core::model::GraphModel;
use serde_json::{json, Value};

const ARTIFACT: &str = include_str!("fixtures/graph_model_v1.json");

fn artifact() -> Value {
    serde_json::from_str(ARTIFACT).unwrap()
}

fn rejection(mutate: impl FnOnce(&mut Value)) -> String {
    let mut value = artifact();
    mutate(&mut value);
    match GraphModel::from_json(&value.to_string()) {
        Ok(_) => panic!("an unknown field was accepted and silently dropped"),
        Err(error) => error.to_string(),
    }
}

/// The message must name the offending field, or a corrupted artifact turns
/// into an unattributable "invalid artifact".
fn assert_names(message: &str, field: &str) {
    assert!(
        message.contains("unknown field") && message.contains(field),
        "message did not name `{field}`: {message}"
    );
}

#[test]
fn an_unknown_top_level_field_is_rejected() {
    let message = rejection(|a| a["provenance"] = json!("some newer writer"));
    assert_names(&message, "provenance");
}

#[test]
fn an_unknown_field_inside_the_schema_is_rejected() {
    // The reported case: this survived `from_json` and vanished on `to_json`.
    let message = rejection(|a| a["schema"]["scalars"] = json!([]));
    assert_names(&message, "scalars");

    let message = rejection(|a| a["schema"]["matrices"][0]["stride"] = json!(2));
    assert_names(&message, "stride");

    let message = rejection(|a| a["schema"]["matrices"][0]["kind"]["Matrix"]["n_rows"] = json!(7));
    assert_names(&message, "n_rows");
}

#[test]
fn an_unknown_field_inside_a_nested_prior_is_rejected() {
    let message = rejection(|a| a["definition"]["priors"][0]["Normal"]["tau"] = json!(1.0));
    assert_names(&message, "tau");

    let message = rejection(|a| a["definition"]["priors"][1]["VectorNormal"]["n_cols"] = json!(2));
    assert_names(&message, "n_cols");
}

#[test]
fn unknown_fields_elsewhere_in_the_definition_are_rejected() {
    let message = rejection(|a| a["definition"]["constraints"] = json!({}));
    assert_names(&message, "constraints");

    let message = rejection(|a| a["definition"]["likelihoods"][0]["weights"] = json!("w"));
    assert_names(&message, "weights");

    let message = rejection(|a| {
        a["definition"]["likelihoods"][0]["mu_expr"]["Add"][1]["MatVec"]["scale"] = json!(1.0)
    });
    assert_names(&message, "scale");

    // Data bound at build time is dropped on the wire, so a writer that emits
    // it is asking for something this format cannot carry.
    let message = rejection(|a| a["definition"]["bound_data_1d"] = json!({"y": [1.0]}));
    assert_names(&message, "bound_data_1d");
}

#[test]
fn the_shipped_v1_fixture_still_loads_and_round_trips() {
    let model = GraphModel::from_json(ARTIFACT).unwrap();
    let encoded = model.to_json().unwrap();
    let reloaded = GraphModel::from_json(&encoded).unwrap();
    assert_eq!(encoded, reloaded.to_json().unwrap());
    assert_eq!(model.likelihood_names, reloaded.likelihood_names);
    assert_eq!(model.structure.schema, reloaded.structure.schema);

    // Nothing the fixture declares - format, version, priors, likelihoods,
    // potentials, deterministics, dimensions or schema - is lost on the way
    // through the strict decode.
    let decoded: Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, artifact());
}

// ---------------------------------------------------------------------------
// Structural model and structural fit artifacts.
// ---------------------------------------------------------------------------

use rustmc_core::structural::{
    self, Component, SamplingConfig, StructuralConfig, StructuralPosterior, VarianceParameter,
};

/// Inverse-gamma on both variances so the struct-variant arm of
/// `VarianceParameter` - the only arm `deny_unknown_fields` can act on - is
/// actually present in the encoded artifact.
fn structural_config() -> StructuralConfig {
    StructuralConfig {
        components: vec![Component::level(
            "lvl".into(),
            VarianceParameter::InverseGamma {
                shape: 2.0,
                scale: 1.0,
            },
            0.0,
            1.0,
        )],
        observation_variance: VarianceParameter::InverseGamma {
            shape: 3.0,
            scale: 1.0,
        },
        student_df: None,
    }
}

fn structural_posterior() -> StructuralPosterior {
    structural::fit(
        &[0.1, 0.2, 0.15, 0.3],
        None,
        &structural_config(),
        &SamplingConfig {
            chains: 1,
            draws: 4,
            warmup: 4,
            thinning: 1,
            seed: 1,
            store_states: true,
        },
    )
    .unwrap()
}

fn mutated(encoded: &str, mutate: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(encoded).unwrap();
    mutate(&mut value);
    value.to_string()
}

#[test]
fn a_structural_model_artifact_rejects_unknown_fields() {
    let encoded = structural_config().to_json().unwrap();
    let rejection =
        |mutate: fn(&mut Value)| match StructuralConfig::from_json(&mutated(&encoded, mutate)) {
            Ok(_) => panic!("an unknown field was accepted and silently dropped"),
            Err(error) => error.to_string(),
        };

    assert_names(
        &rejection(|a| a["provenance"] = json!("newer writer")),
        "provenance",
    );
    assert_names(
        &rejection(|a| a["model"]["constraints"] = json!({})),
        "constraints",
    );
    assert_names(
        &rejection(|a| a["model"]["components"][0]["damping"] = json!(0.9)),
        "damping",
    );
    assert_names(
        &rejection(|a| a["model"]["observation_variance"]["InverseGamma"]["rate"] = json!(1.0)),
        "rate",
    );
}

#[test]
fn a_structural_fit_artifact_rejects_unknown_fields() {
    let encoded = structural_posterior().to_json().unwrap();
    let rejection =
        |mutate: fn(&mut Value)| match StructuralPosterior::from_json(&mutated(&encoded, mutate)) {
            Ok(_) => panic!("an unknown field was accepted and silently dropped"),
            Err(error) => error.to_string(),
        };

    // The reported case: both of these survived `from_json` and vanished on
    // the next `to_json`.
    assert_names(
        &rejection(|a| a["totally_unknown_field"] = json!(123)),
        "totally_unknown_field",
    );
    assert_names(
        &rejection(|a| a["posterior"]["notes"] = json!("hi")),
        "notes",
    );

    assert_names(
        &rejection(|a| a["posterior"]["config"]["student_scale"] = json!(1.0)),
        "student_scale",
    );
    assert_names(
        &rejection(|a| a["posterior"]["config"]["components"][0]["damping"] = json!(0.9)),
        "damping",
    );
    assert_names(
        &rejection(|a| a["posterior"]["chains"][0][0]["log_density"] = json!(-1.0)),
        "log_density",
    );
}

#[test]
fn structural_artifacts_still_round_trip() {
    let config = structural_config();
    let encoded = config.to_json().unwrap();
    let reloaded = StructuralConfig::from_json(&encoded).unwrap();
    assert_eq!(encoded, reloaded.to_json().unwrap());

    let posterior = structural_posterior();
    let encoded = posterior.to_json().unwrap();
    let reloaded = StructuralPosterior::from_json(&encoded).unwrap();
    assert_eq!(encoded, reloaded.to_json().unwrap());
}

// ---------------------------------------------------------------------------
// Dynamic GLM fit artifact.
// ---------------------------------------------------------------------------

use rustmc_core::dynamic_glm::{fit_dynamic_glm, DynamicGlmConfig, DynamicGlmPosterior, Family};

fn dynamic_glm_observations() -> Vec<Vec<f64>> {
    vec![vec![0., 1., 4., f64::NAN, 2.]]
}

fn dynamic_glm_artifact() -> String {
    let config = DynamicGlmConfig {
        family: Family::Poisson,
        chains: 1,
        draws: 4,
        warmup: 4,
        ..Default::default()
    };
    let y = dynamic_glm_observations();
    fit_dynamic_glm(&y, None, None, &config)
        .unwrap()
        .to_json(&y)
        .unwrap()
}

#[test]
fn a_dynamic_glm_fit_artifact_rejects_unknown_fields() {
    let encoded = dynamic_glm_artifact();
    let rejection =
        |mutate: fn(&mut Value)| match DynamicGlmPosterior::from_json(&mutated(&encoded, mutate)) {
            Ok(_) => panic!("an unknown field was accepted and silently dropped"),
            Err(error) => error.to_string(),
        };

    // This artifact carries no `format` discriminator, so a writer that adds
    // one is describing a format this reader cannot claim to understand.
    assert_names(
        &rejection(|a| a["format"] = json!("rustmc.dynamic-glm")),
        "format",
    );
    assert_names(
        &rejection(|a| a["posterior"]["notes"] = json!("hi")),
        "notes",
    );
    assert_names(
        &rejection(|a| a["posterior"]["config"]["offset"] = json!(1.0)),
        "offset",
    );
    assert_names(
        &rejection(|a| a["posterior"]["chains"][0][0]["log_density"] = json!(-1.0)),
        "log_density",
    );
}

#[test]
fn a_dynamic_glm_fit_artifact_still_round_trips() {
    let encoded = dynamic_glm_artifact();
    let (posterior, observations) = DynamicGlmPosterior::from_json(&encoded).unwrap();
    assert_eq!(encoded, posterior.to_json(&observations).unwrap());

    // Missing observations survive the null encoding as NaN, not as zeros.
    assert!(observations[0][3].is_nan());
}
