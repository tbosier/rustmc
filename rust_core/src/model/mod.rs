//! Declarative graph models shared by Python and standalone Rust applications.
//!
//! Split by stage: `spec` holds the definition types, `validate` the checks
//! a definition must pass, `compile` the lowering to a `Graph`, `native`
//! the model artifact and the Rust-native model, and `fit` a fitted model
//! and what is computed from its draws. The public items are re-exported
//! here by name, so `crate::model::X` paths are unchanged; the helpers the
//! stages share with each other are re-exported at crate visibility only.

mod compile;
mod fit;
mod fit_artifact;
mod native;
mod spec;
mod validate;

pub use compile::compile;
pub use fit::{DeterministicDraws, ModelFit, Prediction, PredictiveDraws};
pub use native::{sample_model_batch, GraphModel, ModelArtifact, ModelBatchCell};
pub use spec::{
    CompiledDefinition, Data1d, Data2d, DisplayParamSpec, HyperParam, LikelihoodFamily,
    LikelihoodSpec, ModelError, ModelResult, ModelSpec, MuExpr, PriorSpec, SigmaSpec,
};
pub use validate::{
    prior_name, reject_discrete_priors_for_gradient_sampling,
    reject_potentials_for_prior_predictive, template_data_for_spec, validate_finite,
    validate_model_references, validate_positive_finite,
};

// Lowering and validation helpers the stages share with each other and with
// `prior_sampling`. They are not API: they take raw lowering state and change
// with the compiler.
pub(crate) use compile::{
    collect_matvec_params, derive_display_draw, display_sample_result, should_auto_noncenter,
};
pub(crate) use native::bind_prediction;
pub(crate) use spec::collect_expr_param_names;
// Shared between the stage modules only, which reach them through their
// `use super::*`.
use fit::constrained_draw_to_raw;
use spec::{param_error, LinearTerms};
pub(crate) use validate::{
    resolve_hyper, resolve_hyper_value, validate_binary_observations, validate_count_observations,
    validate_integer_observations, validate_positive_observations,
};
