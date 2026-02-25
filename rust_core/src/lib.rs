pub mod autodiff;
pub mod distributions;
pub mod graph;
pub mod hmc;
pub mod progress;
pub mod sampler;

// Future: GPU-accelerated log-probability evaluation via wgpu.
//
// Future: Large hierarchical model optimizations — block-diagonal mass
// matrices, group-level parameter vectorization, and sparse graph evaluation.
