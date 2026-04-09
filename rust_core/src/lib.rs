pub mod autodiff;
pub mod compiled_model;
pub mod diagnostics;
pub mod distributions;
pub mod graph;
pub mod hmc;
pub mod mass_matrix;
pub mod nuts;
pub mod progress;
pub mod sampler;

pub use compiled_model::{
    ArtifactError, CompiledModelArtifact, CompiledModelRuntime, ModelMetadata, ModelStep,
    NodeRef, ParameterBlock, SerializableObsFamily, SerializableParamTransform,
};

// Future: GPU-accelerated log-probability evaluation via wgpu.
//
// Future: Large hierarchical model optimizations — richer block structures
// for very large correlated groups and sparse graph evaluation.
