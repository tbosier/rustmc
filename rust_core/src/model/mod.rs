//! Declarative graph models shared by Python and standalone Rust applications.
//!
//! Split by stage: `spec` holds the definition types, `validate` the checks
//! a definition must pass, `compile` the lowering to a `Graph`, and
//! `native` the artifact format and the Rust-native model and fit. Every
//! public item is re-exported here, so `crate::model::X` paths are unchanged.

mod compile;
mod native;
mod spec;
mod validate;

pub use compile::*;
pub use native::*;
pub use spec::*;
pub use validate::*;
