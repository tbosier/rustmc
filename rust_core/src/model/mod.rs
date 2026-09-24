//! Declarative graph models shared by Python and standalone Rust applications.
//!
//! Split by stage: `spec` holds the definition types, `validate` the checks
//! a definition must pass, `compile` the lowering to a `Graph`, `native`
//! the model artifact and the Rust-native model, and `fit` a fitted model
//! and what is computed from its draws. Every
//! public item is re-exported here, so `crate::model::X` paths are unchanged.

mod compile;
mod fit;
mod fit_artifact;
mod native;
mod spec;
mod validate;

pub use compile::*;
pub use fit::*;
pub use native::*;
pub use spec::*;
pub use validate::*;
