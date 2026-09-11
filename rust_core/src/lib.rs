pub mod autodiff;
pub mod bayesian_ar;
pub mod bayesian_forecast;
pub mod bayesian_regression;
pub mod bayesian_seasonal;
pub mod bayesian_trend;
#[path = "legacy/compiled_model.rs"]
pub mod compiled_model;
pub mod data;
pub mod diagnostics;
pub mod distributions;
pub mod dynamic_glm;
pub mod elliptical_slice;
pub mod forecast_batch;
pub mod forecast_diagnostics;
pub mod graph;
pub mod hierarchical;
pub mod hmc;
pub mod hurdle;
pub mod mass_matrix;
pub mod model;
pub mod nuts;
pub mod observation;
pub mod param_ref;
pub mod progress;
pub mod runoff;
pub mod sampler;
pub mod state_space;
pub mod structural;
pub mod target;

pub use compiled_model::{
    ArtifactError, CompiledModelArtifact, CompiledModelRuntime, ModelMetadata, ModelStep, NodeRef,
    ParameterBlock, SerializableObsFamily, SerializableParamTransform,
};
pub use data::{BindError, DataBinding, DataInputs, DataSchema, DataSlot, MatrixBinding, SlotKind};

// Future: GPU-accelerated log-probability evaluation via wgpu.
//
// Future: Large hierarchical model optimizations — richer block structures
// for very large correlated groups and sparse graph evaluation.
