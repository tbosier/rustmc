//! An artifact from a newer or corrupted writer must be rejected, not silently
//! truncated. Serde drops unknown fields by default, and the loader's
//! `compiled.graph.schema != artifact.schema` cross-check runs *after* that, so
//! anything serde discarded is invisible to it.
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
