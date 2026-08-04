pub mod autodiff;
pub mod bayesian_ar;
pub mod bayesian_forecast;
pub mod bayesian_seasonal;
pub mod bayesian_trend;
pub mod compiled_model;
pub mod data;
pub mod diagnostics;
pub mod distributions;
pub mod graph;
pub mod hierarchical;
pub mod hmc;
pub mod mass_matrix;
pub mod nuts;
pub mod param_ref;
pub mod progress;
pub mod sampler;
pub mod state_space;

pub use compiled_model::{
    ArtifactError, CompiledModelArtifact, CompiledModelRuntime, ModelMetadata, ModelStep, NodeRef,
    ParameterBlock, SerializableObsFamily, SerializableParamTransform,
};
pub use data::{BindError, DataBinding, DataInputs, DataSchema, DataSlot, MatrixBinding, SlotKind};

// Future: GPU-accelerated log-probability evaluation via wgpu.
//
// Future: Large hierarchical model optimizations — richer block structures
// for very large correlated groups and sparse graph evaluation.
