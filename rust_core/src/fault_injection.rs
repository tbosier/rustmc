//! Deliberate sampler faults, for showing that the exactness tests detect them.
//!
//! This module exists only with the `fault-injection` cargo feature, which is
//! off by default; without it the hooks below are compiled out and the
//! kernels are unchanged. With it, the hooks are inert until a fault is
//! selected with [`set`], so a build with every feature enabled still samples
//! correctly. The selected fault is process-wide: a test that sets one must
//! not share its process with tests that expect a correct sampler (the
//! exactness power tests live in their own test binary and hold a lock).
//!
//! Not part of the supported API. Never enable the feature outside tests.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};

/// A fault the NUTS kernel can be told to commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Fault {
    /// Sample correctly.
    None = 0,
    /// Biased multinomial merge: every merge of two subtrees, and every
    /// doubling, takes the proposal from the newer subtree, whatever the
    /// weights say. The draw becomes the last point of the trajectory's final
    /// accepted doubling instead of a weighted choice along it.
    FavourNewSubtree = 1,
    /// After warmup, integrate with a step size [`OVERSTEP_FACTOR`] times the
    /// adapted one and weight every non-divergent leaf equally, ignoring its
    /// energy error: the Metropolis correction is skipped.
    OverstepWithoutMetropolis = 2,
}

/// How much larger than adapted the step size is under
/// [`Fault::OverstepWithoutMetropolis`].
pub const OVERSTEP_FACTOR: f64 = 1.3;

static ACTIVE: AtomicU8 = AtomicU8::new(Fault::None as u8);

thread_local! {
    /// Whether the transition running on this thread skips the Metropolis
    /// weighting. Set at the start of every transition and read while its
    /// tree is built; tree building never yields to rayon, so no other
    /// chain's transition can run on this thread in between.
    static SKIP_METROPOLIS: Cell<bool> = const { Cell::new(false) };
}

/// Select the fault every sampler in this process commits from now on.
pub fn set(fault: Fault) {
    ACTIVE.store(fault as u8, Ordering::SeqCst);
}

/// The fault currently selected.
pub fn active() -> Fault {
    match ACTIVE.load(Ordering::SeqCst) {
        1 => Fault::FavourNewSubtree,
        2 => Fault::OverstepWithoutMetropolis,
        _ => Fault::None,
    }
}

/// Whether a multinomial merge should take the newer subtree regardless of
/// the draw that decided it.
pub(crate) fn favours_new_subtree() -> bool {
    active() == Fault::FavourNewSubtree
}

/// Called at the start of every NUTS transition with the step size the
/// kernel would use; returns the step size to integrate with and records
/// whether this transition skips the Metropolis weighting.
pub(crate) fn begin_transition(step_size: f64, is_warmup: bool) -> f64 {
    let skip = !is_warmup && active() == Fault::OverstepWithoutMetropolis;
    SKIP_METROPOLIS.with(|flag| flag.set(skip));
    if skip {
        step_size * OVERSTEP_FACTOR
    } else {
        step_size
    }
}

/// The multinomial log weight of a leaf: `log_weight` unchanged, or zero for
/// a non-divergent leaf when this transition skips the Metropolis weighting.
pub(crate) fn leaf_log_weight(log_weight: f64) -> f64 {
    if log_weight != f64::NEG_INFINITY && SKIP_METROPOLIS.with(Cell::get) {
        0.0
    } else {
        log_weight
    }
}
