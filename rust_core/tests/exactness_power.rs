//! Power controls for the exactness gate in `exactness.rs`: the same gate,
//! at the same shipped defaults, against NUTS kernels broken on purpose.
//! Each fault has to make the gate fail on a moment check (the mean or the
//! variance of some coordinate, or the squared norm of the whitened draws,
//! a finite error of 4 MCSE or more), not merely on the ESS floor or a
//! divergence. The overstep must be caught on all 18 fits (six cases, three
//! seeds each); over 200 more seeds per case its smallest squared-norm error
//! was 8.9 MCSE. The biased merge must be caught on at least 15 of the 18:
//! over 200 seeds per case it was caught in 94% to 100% of fits, so
//! requiring all 18 would turn this test red on about 37% of changes that
//! move the draws with nothing wrong, while at least 15 does so on about
//! 0.06%. A correct sampler fails the gate on about 2% of fits, so it cannot
//! reach 15.
//!
//! The faults come from `rustmc_core::fault_injection`, compiled in only with
//! the `fault-injection` feature:
//!
//! ```text
//! cargo test --release -p rustmc_core --features fault-injection --test exactness_power
//! ```
//!
//! The selected fault is process-wide, so every test here takes [`LOCK`] and
//! restores the correct sampler when it finishes, even by panicking. Nothing
//! else runs in this test binary.
#![cfg(feature = "fault-injection")]

#[path = "exactness_support/mod.rs"]
mod exactness_support;

use exactness_support::{check, fit, gate_cases, label, Outcome, MAX_MCSE};
use rayon::prelude::*;
use rustmc_core::fault_injection::{self, Fault};
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

/// Seeds per case; the gate's own seed and the ones after it.
const SEEDS: u64 = 3;

/// Selects a fault for its lifetime and restores the correct sampler on drop.
struct Injected<'a> {
    _lock: std::sync::MutexGuard<'a, ()>,
}

impl Injected<'_> {
    fn new(fault: Fault) -> Self {
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        fault_injection::set(fault);
        Self { _lock: lock }
    }
}

impl Drop for Injected<'_> {
    fn drop(&mut self) {
        fault_injection::set(Fault::None);
    }
}

/// Every gate case and seed under `fault`.
fn outcomes(fault: Fault) -> Vec<Outcome> {
    let _injected = Injected::new(fault);
    let mut outcomes = Vec::new();
    for (target, path, gate_seed) in gate_cases() {
        for seed in gate_seed..gate_seed + SEEDS {
            let outcome = check(
                &format!("{:?}: {}, seed {seed}", fault, label(&target, path)),
                &fit(&target, path, seed),
                &target,
            );
            println!("{}", outcome.summary());
            outcomes.push(outcome);
        }
    }
    outcomes
}

/// Whether some finite standardized error reached 4 MCSE. A NaN error also
/// fails the gate, but it would not show that the fault moved the draws, so
/// it does not count here.
fn failed_a_moment_check(outcome: &Outcome) -> bool {
    let detected = |z: &f64| z.is_finite() && z.abs() >= MAX_MCSE;
    outcome.mean_z.iter().any(detected)
        || outcome.variance_z.iter().any(detected)
        || detected(&outcome.squared_norm_z)
}

/// Require the gate to catch `fault` on its moments in at least
/// `min_caught` of the fits [`outcomes`] runs.
fn assert_caught(fault: Fault, min_caught: usize) {
    let outcomes = outcomes(fault);
    let missed: Vec<String> = outcomes
        .iter()
        .filter(|o| !failed_a_moment_check(o))
        .map(Outcome::summary)
        .collect();
    let caught = outcomes.len() - missed.len();
    println!("{fault:?}: caught on {caught} of {} fits", outcomes.len());
    assert!(
        caught >= min_caught,
        "the gate caught {fault:?} on its moments in only {caught} of {} fits (at least \
         {min_caught} required); missed:\n{}",
        outcomes.len(),
        missed.join("\n")
    );
}

/// Without a fault the hooks are inert: the gate cases pass at their gate
/// seeds in this build too, so the failures below come from the faults.
#[test]
fn the_gate_passes_with_the_hooks_compiled_in_and_no_fault_selected() {
    let _injected = Injected::new(Fault::None);
    for (target, path, seed) in gate_cases() {
        let outcome = check(&label(&target, path), &fit(&target, path, seed), &target);
        println!("{}", outcome.summary());
        outcome.assert_passed();
    }
}

#[test]
fn the_gate_fails_when_multinomial_merges_favour_the_new_subtree() {
    assert_caught(Fault::FavourNewSubtree, 15);
}

#[test]
fn the_gate_fails_when_the_step_size_is_thirty_percent_too_large_and_unweighted() {
    assert_caught(Fault::OverstepWithoutMetropolis, 18);
}

/// How often each fault is caught, over many seeds, and by which check.
///
/// Runs every gate case under each fault for `RUSTMC_EXACTNESS_SEEDS` seeds
/// (default 50, from 20,000 up) and prints, per fault and case, the fits
/// that failed a moment check, how many of those the per-coordinate checks
/// alone caught, and the median and smallest squared-norm error.
///
/// ```text
/// RUSTMC_EXACTNESS_SEEDS=200 cargo test --release -p rustmc_core \
///     --features fault-injection --test exactness_power -- --ignored --nocapture
/// ```
#[test]
#[ignore = "study: hundreds of fits; run by hand, see the doc comment"]
fn detection_rate_over_seeds() {
    let seeds: u64 = std::env::var("RUSTMC_EXACTNESS_SEEDS").map_or(50, |v| {
        v.parse()
            .expect("RUSTMC_EXACTNESS_SEEDS must be a whole number")
    });
    let started = std::time::Instant::now();
    for fault in [Fault::FavourNewSubtree, Fault::OverstepWithoutMetropolis] {
        let _injected = Injected::new(fault);
        for (target, path, _) in gate_cases() {
            let outcomes: Vec<Outcome> = (20_000..20_000 + seeds)
                .into_par_iter()
                .map(|seed| check(&label(&target, path), &fit(&target, path, seed), &target))
                .collect();
            let caught = outcomes.iter().filter(|o| failed_a_moment_check(o)).count();
            let per_coordinate = outcomes
                .iter()
                .filter(|o| {
                    o.mean_z
                        .iter()
                        .chain(&o.variance_z)
                        .any(|z| z.is_finite() && z.abs() >= MAX_MCSE)
                })
                .count();
            let mut norm: Vec<f64> = outcomes.iter().map(|o| o.squared_norm_z.abs()).collect();
            norm.sort_by(f64::total_cmp);
            println!(
                "{fault:?}, {}: caught {caught}/{seeds}; per-coordinate checks alone \
                 {per_coordinate}/{seeds}; squared-norm error size median {:.2}, smallest {:.2} MCSE",
                label(&target, path),
                norm[norm.len() / 2],
                norm[0]
            );
        }
    }
    println!("{:.1} s", started.elapsed().as_secs_f64());
}
