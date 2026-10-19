//! Sampler exactness at the shipped defaults, on targets whose moments are
//! known in closed form.
//!
//! The recovery suite compares posterior means with the parameters that
//! generated the data, under hand-picked tolerances. That cannot see a small
//! sampler bias: the data-generating value is not the posterior mean, and a
//! tolerance wide enough for that gap also hides a bias of a few tenths of a
//! posterior standard deviation. Here the target *is* the posterior, so the
//! only error left is Monte Carlo error, and every check is scaled by the
//! sampler's own Monte Carlo standard error (MCSE) for that coordinate.
//!
//! Every run uses the shipped defaults (`SamplerConfig::default()`: NUTS,
//! 4 chains, 500 warmup, 1000 draws, target acceptance 0.80, metric auto,
//! tree depth 10) apart from the seed, and is checked per coordinate as
//! `exactness_support` describes: no divergences, bulk ESS at least 400, mean
//! within 4 MCSE, and variance within 4 MCSE of the squared deviations; and
//! once per fit, the mean squared norm of the whitened draws within 4 MCSE of
//! the dimension.
//!
//! The targets, each through the paths it is run on:
//!
//! - 100 independent normals, through `target::sample_target` (custom target,
//!   diagonal metric) and through a graph (one vector parameter, where the
//!   auto metric may choose a dense block);
//! - a 50-dimensional Gaussian with AR(1) correlation 0.95 (correlation
//!   matrix condition number 989, covariance 1481), through both paths, and
//!   the same with correlation 0.8 (condition numbers 77 and 116) through the
//!   custom-target path, the case this gate started with;
//! - a 10-coefficient Bayesian linear regression with known noise, whose
//!   posterior is computed in closed form, through a graph written as a user
//!   would write it.
//!
//! `exactness_power.rs` runs the same gate against deliberately broken NUTS
//! kernels (the `fault-injection` feature) and requires it to fail.
//!
//! The ignored `false_alarm_rate_over_seeds` measures how often the correct
//! sampler fails the gate, and `exactness_cost_report` writes the leapfrog
//! steps per bulk ESS, warmup included, as JSON. Both are described, with
//! their results, in docs/statistical-validation.md.
//!
//! Static HMC does not pass, and its two cases (the independent normals and
//! the rho 0.8 correlated Gaussian) are ignored reproducers rather than
//! checks loosened until they pass. The ignored tests print what they find,
//! and docs/statistical-validation.md quotes those runs.
#[path = "exactness_support/mod.rs"]
mod exactness_support;

use exactness_support::{
    check, default_config, fit, fit_with, gate_cases, label, nominal_false_alarm_rate,
    with_sampler, within, Check, GaussianTarget, Outcome, Path, MAX_MCSE, NOMINAL_EXCEEDANCE,
};
use rayon::prelude::*;
use rustmc_core::mass_matrix::MetricKind;
use rustmc_core::sampler::{SamplerConfig, SamplerType};
use std::collections::BTreeMap;

fn gate(target: GaussianTarget, path: Path, seed: u64) {
    let outcome = check(&label(&target, path), &fit(&target, path, seed), &target);
    println!("{}", outcome.summary());
    outcome.assert_passed();
}

#[test]
fn nuts_matches_the_exact_moments_of_a_hundred_independent_normals() {
    gate(GaussianTarget::independent_normals(100), Path::Custom, 2101);
}

#[test]
fn nuts_matches_the_exact_moments_of_a_hundred_independent_normals_in_a_graph() {
    gate(GaussianTarget::independent_normals(100), Path::Graph, 2111);
}

#[test]
fn nuts_matches_the_exact_moments_of_a_correlated_fifty_dimensional_gaussian() {
    gate(GaussianTarget::correlated(50, 0.8), Path::Custom, 2201);
}

#[test]
fn nuts_matches_the_exact_moments_of_an_ill_conditioned_fifty_dimensional_gaussian() {
    gate(GaussianTarget::correlated(50, 0.95), Path::Custom, 2201);
}

#[test]
fn nuts_matches_the_exact_moments_of_an_ill_conditioned_fifty_dimensional_gaussian_in_a_graph() {
    gate(GaussianTarget::correlated(50, 0.95), Path::Graph, 2211);
}

#[test]
fn nuts_matches_the_exact_posterior_of_a_conjugate_regression() {
    gate(GaussianTarget::conjugate_regression(), Path::Graph, 2301);
}

/// The study and the cost report below take their cases from `gate_cases`;
/// this keeps that list identical to the tests above.
#[test]
fn the_gate_cases_are_the_tested_ones() {
    let cases: Vec<(String, u64)> = gate_cases()
        .iter()
        .map(|(target, path, seed)| (label(target, *path), *seed))
        .collect();
    let expected = [
        (
            "independent normals (100 dimensions), custom target path",
            2101,
        ),
        ("independent normals (100 dimensions), graph path", 2111),
        (
            "correlated Gaussian, rho 0.8 (50 dimensions), custom target path",
            2201,
        ),
        (
            "correlated Gaussian, rho 0.95 (50 dimensions), custom target path",
            2201,
        ),
        (
            "correlated Gaussian, rho 0.95 (50 dimensions), graph path",
            2211,
        ),
        ("conjugate regression (10 dimensions), graph path", 2301),
    ]
    .map(|(l, s)| (l.to_string(), s));
    assert_eq!(cases, expected);
}

/// Reproducer for an open finding (see the module documentation): at the
/// default 15 leapfrog steps, static HMC's integration time lands near a full
/// period of these Gaussians and some coordinates barely move.
#[test]
#[ignore = "open finding: default static HMC fails the gate on Gaussian targets"]
fn hmc_matches_the_exact_moments_of_a_hundred_independent_normals() {
    let target = GaussianTarget::independent_normals(100);
    let fit = fit_with(&target, Path::Custom, with_sampler(2102, SamplerType::Hmc));
    let outcome = check("HMC, 100 independent normals", &fit, &target);
    println!("{}", outcome.summary());
    outcome.assert_passed();
}

/// Reproducer for the same open finding on the rho 0.8 correlated target.
#[test]
#[ignore = "open finding: default static HMC fails the gate on Gaussian targets"]
fn hmc_matches_the_exact_moments_of_a_correlated_fifty_dimensional_gaussian() {
    let target = GaussianTarget::correlated(50, 0.8);
    let fit = fit_with(&target, Path::Custom, with_sampler(2202, SamplerType::Hmc));
    let outcome = check("HMC, 50 correlated Gaussian coordinates", &fit, &target);
    println!("{}", outcome.summary());
    outcome.assert_passed();
}

/// What the auto metric buys on the ill-conditioned target's graph path,
/// against a diagonal and a dense metric, at the shipped defaults otherwise.
/// Prints each fit's summary for three seeds per metric.
///
/// ```text
/// cargo test --release -p rustmc_core --test exactness -- --ignored --nocapture \
///     metric_choice_on_the_ill_conditioned_graph_target
/// ```
#[test]
#[ignore = "measurement: prints the cost of each metric; run by hand"]
fn metric_choice_on_the_ill_conditioned_graph_target() {
    let target = GaussianTarget::correlated(50, 0.95);
    for metric in [MetricKind::Auto, MetricKind::Diagonal, MetricKind::Dense] {
        for seed in 2211..2214 {
            let config = SamplerConfig {
                metric,
                ..default_config(seed)
            };
            let outcome = check(
                &format!("{metric:?}, seed {seed}"),
                &fit_with(&target, Path::Graph, config),
                &target,
            );
            println!("{}", outcome.summary());
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{name} must be a whole number, not {v:?}"))
    })
}

/// Fraction of `values` whose magnitude is at least `bound` (NaN counts).
fn exceedance(values: &[f64], bound: f64) -> f64 {
    values.iter().filter(|&&z| !within(z, bound)).count() as f64 / values.len() as f64
}

fn standard_deviation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

/// The false-alarm rate of the gate on the correct sampler.
///
/// Runs every gate case at the shipped defaults over `RUSTMC_EXACTNESS_SEEDS`
/// seeds (default 200, from 10,000 up, none of them a gate seed) and prints,
/// per case: how many fits failed the gate and on which check, against the
/// rate the moment checks alone would have if every standardized error were
/// an independent standard normal (two checks per coordinate and one on the
/// squared norm, each crossing 4 MCSE with probability 6.3e-5); and, pooled
/// over coordinates and seeds,
/// the fraction of standardized errors beyond 2, 3 and 4 MCSE and their
/// standard deviation, which a correct MCSE puts near 4.55%, 0.27%, 0.0063%
/// and 1. With `RUSTMC_EXACTNESS_JSON` set, also writes the tallies there.
///
/// ```text
/// RUSTMC_EXACTNESS_SEEDS=1000 cargo test --release -p rustmc_core \
///     --test exactness -- --ignored --nocapture false_alarm_rate_over_seeds
/// ```
#[test]
#[ignore = "study: hundreds of fits; run by hand, see the doc comment"]
fn false_alarm_rate_over_seeds() {
    let seeds = env_usize("RUSTMC_EXACTNESS_SEEDS", 200);
    let started = std::time::Instant::now();
    let mut records = Vec::new();
    for (target, path, _) in gate_cases() {
        let case_started = std::time::Instant::now();
        let name = label(&target, path);
        let outcomes: Vec<Outcome> = (0..seeds as u64)
            .into_par_iter()
            .map(|k| {
                let seed = 10_000 + k;
                check(
                    &format!("{name}, seed {seed}"),
                    &fit(&target, path, seed),
                    &target,
                )
            })
            .collect();
        let failed = outcomes.iter().filter(|o| !o.passed()).count();
        let mut by_check: BTreeMap<String, usize> = BTreeMap::new();
        for check in [
            Check::Divergences,
            Check::BulkEss,
            Check::Mean,
            Check::Variance,
            Check::SquaredNorm,
        ] {
            by_check.insert(
                format!("{check:?}"),
                outcomes.iter().filter(|o| o.failed(check)).count(),
            );
        }
        let moment_failures = outcomes
            .iter()
            .filter(|o| {
                o.failed(Check::Mean) || o.failed(Check::Variance) || o.failed(Check::SquaredNorm)
            })
            .count();
        let mean_z: Vec<f64> = outcomes.iter().flat_map(|o| o.mean_z.clone()).collect();
        let variance_z: Vec<f64> = outcomes.iter().flat_map(|o| o.variance_z.clone()).collect();
        let squared_norm_z: Vec<f64> = outcomes.iter().map(|o| o.squared_norm_z).collect();
        let nominal = nominal_false_alarm_rate(target.dimension());
        println!(
            "{name}: {failed}/{seeds} fits failed the gate ({:.2}%); moment checks failed in \
             {moment_failures} ({:.2}%), nominal {:.2}%; by check {by_check:?}",
            100.0 * failed as f64 / seeds as f64,
            100.0 * moment_failures as f64 / seeds as f64,
            100.0 * nominal
        );
        for (which, z) in [
            ("mean", &mean_z),
            ("variance", &variance_z),
            ("squared norm", &squared_norm_z),
        ] {
            println!(
                "    {which} errors: {} values, beyond 2 MCSE {:.3}%, beyond 3 {:.4}%, \
                 beyond 4 {:.4}%, sd {:.3}",
                z.len(),
                100.0 * exceedance(z, 2.0),
                100.0 * exceedance(z, 3.0),
                100.0 * exceedance(z, MAX_MCSE),
                standard_deviation(z)
            );
        }
        for outcome in outcomes.iter().filter(|o| !o.passed()) {
            println!("    {}", outcome.summary());
        }
        let z_summary = |z: &Vec<f64>| {
            serde_json::json!({
                "count": z.len(),
                "fraction_beyond_2": exceedance(z, 2.0),
                "fraction_beyond_3": exceedance(z, 3.0),
                "fraction_beyond_4": exceedance(z, MAX_MCSE),
                "standard_deviation": standard_deviation(z),
            })
        };
        records.push(serde_json::json!({
            "case": name,
            "dimension": target.dimension(),
            "seeds": seeds,
            "first_seed": 10_000,
            "fits_failed": failed,
            "fits_failed_by_check": by_check,
            "fits_failed_on_moments": moment_failures,
            "nominal_moment_false_alarm_rate": nominal,
            "mean_errors": z_summary(&mean_z),
            "variance_errors": z_summary(&variance_z),
            "squared_norm_errors": z_summary(&squared_norm_z),
            "wall_seconds": case_started.elapsed().as_secs_f64(),
        }));
    }
    println!(
        "{} fits in {:.1} s",
        seeds * records.len(),
        started.elapsed().as_secs_f64()
    );
    if let Ok(path) = std::env::var("RUSTMC_EXACTNESS_JSON") {
        let report = serde_json::json!({
            "format": "rustmc.exactness-false-alarm",
            "version": 1,
            "revision": command_output("git", &["rev-parse", "HEAD"]),
            "dirty": command_output(
                "git",
                &[
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    ".",
                    ":(exclude)benchmarks/results",
                ],
            )
            .map(|s| !s.is_empty()),
            "cpu_model": cpu_model(),
            "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
            "command": format!(
                "RUSTMC_EXACTNESS_SEEDS={seeds} RUSTMC_EXACTNESS_JSON=<path> cargo test \
                 --release -p rustmc_core --test exactness -- --ignored --nocapture \
                 false_alarm_rate_over_seeds"
            ),
            "nominal_exceedance_per_check": NOMINAL_EXCEEDANCE,
            "cases": records,
            "wall_seconds": started.elapsed().as_secs_f64(),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap() + "\n").unwrap();
    }
}

/// Leapfrog steps per unit of bulk ESS, warmup included, for every gate case.
///
/// Fits each case at the shipped defaults for `RUSTMC_EXACTNESS_COST_SEEDS`
/// seeds (default 10: the gate's own seed and the ones after it) and writes,
/// to the path in `RUSTMC_EXACTNESS_JSON` (required; a relative path is
/// taken from `rust_core`, where cargo runs the test), each fit's leapfrog
/// steps in warmup and in draws (from the transition records; the step-size
/// searches warmup runs are not transitions and are not counted), its
/// smallest bulk ESS, their ratios, and the machine and command.
///
/// ```text
/// RUSTMC_EXACTNESS_JSON=$PWD/benchmarks/results/<date>-exactness.json \
///     cargo test --release -p rustmc_core --test exactness -- --ignored \
///     --nocapture exactness_cost_report
/// ```
#[test]
#[ignore = "writes a benchmark record; run by hand, see the doc comment"]
fn exactness_cost_report() {
    let output = std::env::var("RUSTMC_EXACTNESS_JSON")
        .expect("set RUSTMC_EXACTNESS_JSON to the file to write");
    let seeds = env_usize("RUSTMC_EXACTNESS_COST_SEEDS", 10) as u64;
    let mut cases = Vec::new();
    for (target, path, gate_seed) in gate_cases() {
        let name = label(&target, path);
        let mut runs = Vec::new();
        for seed in gate_seed..gate_seed + seeds {
            let started = std::time::Instant::now();
            let fit = fit(&target, path, seed);
            let seconds = started.elapsed().as_secs_f64();
            let outcome = check(&name, &fit, &target);
            let total = outcome.warmup_leapfrog_steps + outcome.draw_leapfrog_steps;
            runs.push(serde_json::json!({
                "seed": seed,
                "passed_gate": outcome.passed(),
                "warmup_leapfrog_steps": outcome.warmup_leapfrog_steps,
                "draw_leapfrog_steps": outcome.draw_leapfrog_steps,
                "min_bulk_ess": outcome.min_ess_bulk(),
                "leapfrog_steps_per_min_bulk_ess_with_warmup":
                    total as f64 / outcome.min_ess_bulk(),
                "leapfrog_steps_per_min_bulk_ess_draws_only":
                    outcome.draw_leapfrog_steps as f64 / outcome.min_ess_bulk(),
                "divergences": outcome.divergences,
                "max_r_hat": outcome.max_r_hat,
                "worst_mean_error_mcse": outcome.worst_mean_z(),
                "worst_variance_error_mcse": outcome.worst_variance_z(),
                "squared_norm_error_mcse": outcome.squared_norm_z,
                "fit_wall_seconds": seconds,
            }));
        }
        let mut ratios: Vec<f64> = runs
            .iter()
            .map(|r| {
                r["leapfrog_steps_per_min_bulk_ess_with_warmup"]
                    .as_f64()
                    .unwrap()
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        let middle = ratios.len() / 2;
        let median = if ratios.len() % 2 == 1 {
            ratios[middle]
        } else {
            0.5 * (ratios[middle - 1] + ratios[middle])
        };
        println!(
            "{name}: leapfrog steps per min bulk ESS, warmup included: median {median:.1}, \
             range {:.1} to {:.1} over {seeds} seeds",
            ratios[0],
            ratios[ratios.len() - 1]
        );
        cases.push(serde_json::json!({
            "case": name,
            "target": target.name,
            "dimension": target.dimension(),
            "path": path.name(),
            "median_leapfrog_steps_per_min_bulk_ess_with_warmup": median,
            "runs": runs,
        }));
    }
    let report = serde_json::json!({
        "format": "rustmc.exactness-cost",
        "version": 1,
        "date": command_output("date", &["-u", "+%Y-%m-%d"]),
        "rustmc_core": env!("CARGO_PKG_VERSION"),
        "revision": command_output("git", &["rev-parse", "HEAD"]),
        "dirty": command_output(
                "git",
                &[
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    ".",
                    ":(exclude)benchmarks/results",
                ],
            )
            .map(|s| !s.is_empty()),
        "machine": {
            "cpu_model": cpu_model(),
            "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "kernel": command_output("uname", &["-r"]),
            "rustc": command_output("rustc", &["--version"]),
            "RAYON_NUM_THREADS": std::env::var("RAYON_NUM_THREADS").ok(),
        },
        "command": "RUSTMC_EXACTNESS_JSON=<path> cargo test --release -p rustmc_core \
                    --test exactness -- --ignored --nocapture exactness_cost_report",
        "sampling": {
            "sampler": "NUTS",
            "chains": 4,
            "warmup": 500,
            "draws": 1000,
            "target_accept": 0.8,
            "metric": "auto",
            "max_tree_depth": 10,
            "threads_per_fit": 1,
        },
        "definition": "leapfrog steps recorded in every transition of all four chains, \
                       warmup and draws, divided by the smallest bulk ESS over \
                       coordinates; the step-size searches at the start of warmup and \
                       after each metric update are not transitions and are not counted",
        "not_measured": "other engines (this is a rustmc-only cost record, so the \
                         template's PyMC, nutpie, NumPyro, JAX, ArviZ, NumPy and Python \
                         fields do not apply), compilation and post-processing time \
                         (fit_wall_seconds is the whole sampling call), and peak RSS",
        "data": "every target is generated deterministically in \
                 rust_core/tests/exactness_support/mod.rs; no external data",
        "cases": cases,
    });
    std::fs::write(
        &output,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn cpu_model() -> Option<String> {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .map(|model| model.trim().to_string())
}
