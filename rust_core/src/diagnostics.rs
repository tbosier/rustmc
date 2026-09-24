use crate::hmc::TransitionStats;
use rayon::prelude::*;
use std::collections::HashSet;

/// Narrowest the parameter-name column is ever drawn.
///
/// This is the historical width, so tables whose names all fit keep exactly
/// the layout they have always had.
const NAME_COLUMN_MIN_WIDTH: usize = 12;

/// Widest the parameter-name column is drawn before names are abbreviated.
///
/// A single pathological name (nothing stops a user passing a 200-character
/// one) would otherwise push the numeric columns past the edge of a terminal,
/// where line wrapping destroys the alignment of *every* row. Abbreviating one
/// label is the smaller loss. See `display_names` for the rule that stops
/// abbreviation from ever making two distinct parameters print identically.
const NAME_COLUMN_MAX_WIDTH: usize = 48;

/// MCMC diagnostic computations: R-hat, ESS, MCSE, quantiles.
///
/// All algorithms follow the definitions in:
///   Vehtari et al. (2021) "Rank-normalization, folding, and localization:
///   An improved R-hat for assessing convergence of MCMC"
/// Per-parameter diagnostic summary.
#[derive(Debug, Clone)]
pub struct ParamDiagnostics {
    pub name: String,
    pub mean: f64,
    pub std: f64,
    pub hdi_3: f64,
    pub hdi_97: f64,
    pub ess_bulk: f64,
    pub ess_tail: f64,
    pub r_hat: f64,
    pub mcse_mean: f64,
}

/// Full diagnostic report for a sampling run.
#[derive(Debug, Clone)]
pub struct DiagnosticsReport {
    pub params: Vec<ParamDiagnostics>,
    pub num_chains: usize,
    pub num_draws: usize,
    pub accept_rates: Vec<f64>,
    pub divergences: usize,
}

/// Per-chain transition telemetry summary.
///
/// Divergence and acceptance fields describe posterior draws only. Energy and
/// integrator-work fields summarize every retained transition, including warmup.
#[derive(Debug, Clone)]
pub struct ChainTransitionDiagnostics {
    pub chain_index: usize,
    pub num_transitions: usize,
    pub num_warmup_transitions: usize,
    pub num_draw_transitions: usize,
    pub divergences: usize,
    pub accepted_transitions: usize,
    pub mean_accept_prob: f64,
    pub mean_energy_error: f64,
    pub max_abs_energy_error: f64,
    pub mean_step_size: f64,
    pub max_tree_depth: Option<usize>,
    pub total_leapfrog_steps: usize,
}

/// Aggregated telemetry across all chains for one sampling run.
///
/// `total_divergences` and `mean_accept_prob` describe posterior draws only;
/// energy and leapfrog fields summarize all retained transitions.
#[derive(Debug, Clone)]
pub struct TransitionDiagnosticsReport {
    pub chains: Vec<ChainTransitionDiagnostics>,
    pub total_transitions: usize,
    pub total_warmup_transitions: usize,
    pub total_draw_transitions: usize,
    pub total_divergences: usize,
    pub total_leapfrog_steps: usize,
    pub mean_accept_prob: f64,
    pub mean_energy_error: f64,
    pub max_abs_energy_error: f64,
}

/// Number of `char`s in `text`, the unit this table measures columns in.
///
/// Byte length is the wrong measure: the rules are drawn with `─` and
/// abbreviated names end in `…`, both multi-byte. Counting `char`s is also what
/// `format!`'s own `{:width$}` padding counts, so measurement and padding
/// always agree and the table is internally consistent.
///
/// This is not true terminal-cell width: a full-width or combining character in
/// a parameter name would still render misaligned. Computing that needs a
/// Unicode width table, which is not worth a dependency for parameter names.
fn display_width(text: &str) -> usize {
    text.chars().count()
}

/// Per-column widths: the larger of the minimum, the header and every cell.
fn column_widths<const N: usize>(
    headers: &[&str; N],
    minimums: &[usize; N],
    rows: &[[String; N]],
) -> [usize; N] {
    let mut widths = *minimums;
    for (width, header) in widths.iter_mut().zip(headers.iter()) {
        *width = (*width).max(display_width(header));
    }
    for row in rows {
        for (width, cell) in widths.iter_mut().zip(row.iter()) {
            *width = (*width).max(display_width(cell));
        }
    }
    widths
}

/// Render one row: first column left aligned, the rest right aligned, joined
/// by single spaces.
fn render_row<const N: usize>(cells: &[String; N], widths: &[usize; N]) -> String {
    cells
        .iter()
        .zip(widths.iter())
        .enumerate()
        .map(|(index, (cell, &width))| {
            if index == 0 {
                format!("{cell:<width$}")
            } else {
                format!("{cell:>width$}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The horizontal rule, spanning every column plus the single-space gaps.
fn rule<const N: usize>(widths: &[usize; N]) -> String {
    "─".repeat(widths.iter().sum::<usize>() + N.saturating_sub(1))
}

/// Labels for the parameter-name column.
///
/// Names longer than [`NAME_COLUMN_MAX_WIDTH`] are abbreviated keeping both
/// ends, so indexed names such as `beta[997]` stay distinguishable. If
/// abbreviating would make any two labels print identically, abbreviation is
/// abandoned for the whole table: an over-wide table is recoverable, a table in
/// which two different parameters print the same name is not.
///
/// Labels are compared as they will be *printed*, so trailing spaces are
/// ignored: the column pads every label out to the same width, which would
/// otherwise hide the difference between `foo` and `foo `.
fn display_names(params: &[ParamDiagnostics]) -> Vec<String> {
    let full: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
    if full
        .iter()
        .all(|name| display_width(name) <= NAME_COLUMN_MAX_WIDTH)
    {
        return full;
    }

    let shortened: Vec<String> = full
        .iter()
        .map(|name| abbreviate(name, NAME_COLUMN_MAX_WIDTH))
        .collect();
    let distinct: HashSet<&str> = shortened
        .iter()
        .map(|label| label.trim_end_matches(' '))
        .collect();
    if distinct.len() == shortened.len() {
        shortened
    } else {
        full
    }
}

/// Shorten `text` to `max_width` display columns, keeping its head and tail.
fn abbreviate(text: &str, max_width: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_width {
        return text.to_string();
    }
    let tail = max_width / 3;
    // One column is spent on the ellipsis itself.
    let head = max_width - tail - 1;
    let mut out: String = chars[..head].iter().collect();
    out.push('…');
    out.extend(&chars[chars.len() - tail..]);
    out
}

/// Format an effective-sample-size count, which is NaN when unavailable.
fn count_cell(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.0}")
    } else {
        "NaN".to_string()
    }
}

impl DiagnosticsReport {
    /// Render the diagnostics as a formatted table string.
    pub fn to_table(&self) -> String {
        self.to_table_with_sampler(None)
    }

    /// Supply sampler-specific metadata instead of Hamiltonian telemetry.
    pub fn to_table_with_sampler(&self, sampler: Option<&str>) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{} chains × {} draws per chain",
            self.num_chains, self.num_draws
        ));
        lines.push(String::new());

        const HEADERS: [&str; 9] = [
            "Parameter",
            "mean",
            "std",
            "hdi_3%",
            "hdi_97%",
            "ess_bulk",
            "ess_tail",
            "r_hat",
            "mcse_mean",
        ];
        // The historical fixed widths, now used as minimums. Ordinary tables
        // keep exactly the columns they had; a column whose content would
        // overflow grows instead of shoving every column after it out of
        // alignment. The rules are the one deliberate change even for ordinary
        // tables: they were hard-coded at 96 while the columns only ever summed
        // to 94, and they now span the real width.
        const MIN_WIDTHS: [usize; 9] = [NAME_COLUMN_MIN_WIDTH, 8, 8, 10, 10, 10, 10, 8, 10];

        let rows: Vec<[String; 9]> = self
            .params
            .iter()
            .zip(display_names(&self.params))
            .map(|(p, name)| {
                [
                    name,
                    format!("{:.4}", p.mean),
                    format!("{:.4}", p.std),
                    format!("{:.4}", p.hdi_3),
                    format!("{:.4}", p.hdi_97),
                    count_cell(p.ess_bulk),
                    count_cell(p.ess_tail),
                    format!("{:.4}", p.r_hat),
                    format!("{:.6}", p.mcse_mean),
                ]
            })
            .collect();

        let widths = column_widths(&HEADERS, &MIN_WIDTHS, &rows);
        lines.push(render_row(&HEADERS.map(String::from), &widths));
        lines.push(rule(&widths));
        lines.extend(rows.iter().map(|row| render_row(row, &widths)));
        lines.push(rule(&widths));

        if let Some(sampler) = sampler {
            lines.push(sampler.to_string());
        } else {
            let avg_accept: f64 = if self.accept_rates.is_empty() {
                0.0
            } else {
                self.accept_rates.iter().sum::<f64>() / self.accept_rates.len() as f64
            };
            lines.push(format!(
                "Mean accept rate: {:.2}  │  Divergences: {}",
                avg_accept, self.divergences
            ));
        }

        let any_bad_rhat = self
            .params
            .iter()
            .any(|p| p.r_hat > 1.01 || !p.r_hat.is_finite());
        let any_low_ess = self
            .params
            .iter()
            .any(|p| p.ess_bulk < 400.0 || p.ess_tail < 400.0);

        if any_bad_rhat {
            lines.push(
                "WARNING: Some R-hat values > 1.01; chains may not have converged.".to_string(),
            );
        }
        if any_low_ess {
            lines.push(
                "WARNING: Some ESS values < 400; consider increasing draws or tuning.".to_string(),
            );
        }
        if sampler.is_none() && self.divergences > 0 {
            lines.push(format!(
                "WARNING: {} divergent transitions; results may be unreliable.",
                self.divergences
            ));
        }

        lines.join("\n")
    }
}

impl TransitionDiagnosticsReport {
    /// Render the telemetry as a compact table.
    pub fn to_table(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Transition telemetry: {} transitions ({} warmup, {} draws), {} divergences",
            self.total_transitions,
            self.total_warmup_transitions,
            self.total_draw_transitions,
            self.total_divergences
        ));
        lines.push(String::new());

        const HEADERS: [&str; 9] = [
            "chain",
            "trans",
            "warmup",
            "div",
            "acc",
            "mean_acc",
            "mean_dH",
            "max|dH|",
            "leapfrogs",
        ];
        // As above: historical widths as minimums, so a huge leapfrog count or
        // energy error widens its own column rather than the whole table. This
        // rule was hard-coded at 100 while the columns summed to 96.
        const MIN_WIDTHS: [usize; 9] = [6, 8, 8, 10, 10, 12, 12, 12, 10];

        let rows: Vec<[String; 9]> = self
            .chains
            .iter()
            .map(|chain| {
                [
                    chain.chain_index.to_string(),
                    chain.num_transitions.to_string(),
                    chain.num_warmup_transitions.to_string(),
                    chain.divergences.to_string(),
                    chain.accepted_transitions.to_string(),
                    format!("{:.4}", chain.mean_accept_prob),
                    format!("{:.4}", chain.mean_energy_error),
                    format!("{:.4}", chain.max_abs_energy_error),
                    chain.total_leapfrog_steps.to_string(),
                ]
            })
            .collect();

        let widths = column_widths(&HEADERS, &MIN_WIDTHS, &rows);
        lines.push(render_row(&HEADERS.map(String::from), &widths));
        lines.push(rule(&widths));
        lines.extend(rows.iter().map(|row| render_row(row, &widths)));
        lines.push(rule(&widths));
        lines.push(format!(
            "Mean accept prob: {:.4}  |  Mean dH: {:.4}  |  Max |dH|: {:.4}",
            self.mean_accept_prob, self.mean_energy_error, self.max_abs_energy_error
        ));
        lines.join("\n")
    }
}

/// Compute full diagnostics from samples[chain][draw][param].
pub fn compute_diagnostics(
    samples: &[Vec<Vec<f64>>],
    param_names: &[String],
    accept_rates: &[f64],
    divergences: usize,
) -> DiagnosticsReport {
    let n_chains = samples.len();
    let n_draws = if n_chains > 0 { samples[0].len() } else { 0 };
    let n_params = param_names.len();

    // Validate before rank normalization: ragged arrays cannot preserve chain
    // axes, and empty arrays do not define a sampling distribution.
    if n_chains == 0
        || n_draws == 0
        || samples
            .iter()
            .any(|chain| chain.len() != n_draws || chain.iter().any(|draw| draw.len() != n_params))
    {
        return DiagnosticsReport {
            params: param_names
                .iter()
                .cloned()
                .map(unavailable_parameter)
                .collect(),
            num_chains: n_chains,
            num_draws: n_draws,
            accept_rates: accept_rates.to_vec(),
            divergences,
        };
    }

    // Parameters are independent, and a hierarchical fit can have thousands.
    let params = param_names
        .par_iter()
        .enumerate()
        .map(|(pidx, name)| parameter_diagnostics(samples, pidx, name))
        .collect();

    DiagnosticsReport {
        params,
        num_chains: n_chains,
        num_draws: n_draws,
        accept_rates: accept_rates.to_vec(),
        divergences,
    }
}

fn parameter_diagnostics(samples: &[Vec<Vec<f64>>], pidx: usize, name: &str) -> ParamDiagnostics {
    let chains: Vec<Vec<f64>> = samples
        .iter()
        .map(|chain| chain.iter().map(|draw| draw[pidx]).collect())
        .collect();

    if chains.iter().flatten().any(|value| !value.is_finite()) {
        return unavailable_parameter(name.to_string());
    }
    let (mean, std) = scaled_moments(|| chains.iter().flatten().copied());
    let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (hdi_3, hdi_97) = hdi_interval_sorted(&all, 0.94);
    let ess_mean = ess_raw(&chains);
    let mcse_mean = if ess_mean > 0.0 {
        std / ess_mean.sqrt()
    } else {
        f64::NAN
    };

    ParamDiagnostics {
        name: name.to_string(),
        mean,
        std,
        hdi_3,
        hdi_97,
        ess_bulk: ess_bulk_chains(&chains),
        ess_tail: ess_tail_chains(&chains),
        r_hat: r_hat_chains(&chains),
        mcse_mean,
    }
}

fn unavailable_parameter(name: String) -> ParamDiagnostics {
    ParamDiagnostics {
        name,
        mean: f64::NAN,
        std: f64::NAN,
        hdi_3: f64::NAN,
        hdi_97: f64::NAN,
        ess_bulk: f64::NAN,
        ess_tail: f64::NAN,
        r_hat: f64::NAN,
        mcse_mean: f64::NAN,
    }
}

/// Compute structured transition telemetry from per-chain transition lists.
pub fn compute_transition_diagnostics(
    transitions: &[Vec<TransitionStats>],
) -> TransitionDiagnosticsReport {
    let mut chains = Vec::with_capacity(transitions.len());
    let mut total_transitions = 0usize;
    let mut total_warmup_transitions = 0usize;
    let mut total_draw_transitions = 0usize;
    let mut total_divergences = 0usize;
    let mut total_leapfrog_steps = 0usize;
    let mut sum_accept_prob = 0.0f64;
    let mut sum_energy_error = 0.0f64;
    let mut n_energy_error = 0usize;
    let mut max_abs_energy_error = 0.0f64;

    for (chain_index, chain) in transitions.iter().enumerate() {
        let num_transitions = chain.len();
        let num_warmup_transitions = chain.iter().filter(|t| t.is_warmup).count();
        let num_draw_transitions = num_transitions.saturating_sub(num_warmup_transitions);
        let draws: Vec<&TransitionStats> = chain.iter().filter(|t| !t.is_warmup).collect();
        let divergences = draws.iter().filter(|t| t.divergent).count();
        let accepted_transitions = draws.iter().filter(|t| t.accepted).count();
        let mean_accept_prob = if num_draw_transitions > 0 {
            draws.iter().map(|t| t.accept_prob).sum::<f64>() / num_draw_transitions as f64
        } else {
            0.0
        };
        let mean_energy_error = if num_transitions > 0 {
            chain.iter().map(|t| t.energy_error).sum::<f64>() / num_transitions as f64
        } else {
            0.0
        };
        let chain_max_abs_energy_error = chain
            .iter()
            .map(|t| t.energy_error.abs())
            .fold(0.0, f64::max);
        let mean_step_size = if num_transitions > 0 {
            chain.iter().map(|t| t.step_size).sum::<f64>() / num_transitions as f64
        } else {
            0.0
        };
        let max_tree_depth = chain.iter().filter_map(|t| t.tree_depth).max();
        let chain_leapfrog_steps: usize = chain.iter().map(|t| t.num_leapfrog_steps).sum();

        total_transitions += num_transitions;
        total_warmup_transitions += num_warmup_transitions;
        total_draw_transitions += num_draw_transitions;
        total_divergences += divergences;
        total_leapfrog_steps += chain_leapfrog_steps;
        sum_accept_prob += draws.iter().map(|t| t.accept_prob).sum::<f64>();
        sum_energy_error += chain.iter().map(|t| t.energy_error).sum::<f64>();
        n_energy_error += num_transitions;
        max_abs_energy_error = max_abs_energy_error.max(chain_max_abs_energy_error);

        chains.push(ChainTransitionDiagnostics {
            chain_index,
            num_transitions,
            num_warmup_transitions,
            num_draw_transitions,
            divergences,
            accepted_transitions,
            mean_accept_prob,
            mean_energy_error,
            max_abs_energy_error,
            mean_step_size,
            max_tree_depth,
            total_leapfrog_steps: chain_leapfrog_steps,
        });
    }

    let mean_accept_prob = if total_draw_transitions > 0 {
        sum_accept_prob / total_draw_transitions as f64
    } else {
        0.0
    };
    let mean_energy_error = if n_energy_error > 0 {
        sum_energy_error / n_energy_error as f64
    } else {
        0.0
    };

    TransitionDiagnosticsReport {
        chains,
        total_transitions,
        total_warmup_transitions,
        total_draw_transitions,
        total_divergences,
        total_leapfrog_steps,
        mean_accept_prob,
        mean_energy_error,
        max_abs_energy_error,
    }
}

/// Posterior mean and standard deviation of one parameter's draws, in the
/// draws' own units and without an intermediate that leaves the exponent range.
///
/// This is the single definition of both moments. `compute_diagnostics` uses it
/// for the summary table, and `SampleResult::mean`/`std` and
/// `BatchModelResult::mean`/`std` use it for the values they report directly,
/// so a fit cannot describe its own posterior two different ways.
///
/// `draws` is a factory rather than a slice because the two callers hold the
/// draws in different shapes — chain-major `Vec<Vec<f64>>` per parameter, and a
/// strided read across a `Vec<Vec<Vec<f64>>>` — and neither should have to
/// materialise a copy. It is called five times for ordinary input — once for
/// the first draw, once for the finiteness scan, once for the scale, once for
/// the mean and once for the variance — and six when the centring falls back to
/// zero. It must yield the same sequence, in the same order, every time: the
/// summation order is part of the reported value.
///
/// Draws are centred on the first of them and divided by the largest absolute
/// deviation from it before being summed, so neither the running sum nor
/// `diff * diff` can overflow: every normalised draw lies in `[-1, 1]`. The
/// naive form loses `diff * diff` above `|diff| ~ 1.34e154` and the running sum
/// above `f64::MAX / n`, neither of which is a limit of the posterior. If the
/// deviations themselves are unrepresentable (`-1e308` and `1e308` in one
/// chain) the centring is dropped and the draws are scaled about zero instead.
///
/// The trade the centring makes, which this has always made for the summary
/// table and now makes for the moments a caller reads directly: each
/// normalised draw is rounded once, so the error in the mean is of order one
/// ulp of the draws' *spread* rather than of the mean itself. Draws
/// `[1, -1, 1e-16, 1e-16]` report a mean of 0 where naive summation reports
/// 5e-17. That is 1e-16 of a posterior's spread against a Monte Carlo standard
/// error of order 1e-2 of it, so it is fourteen orders of magnitude below the
/// uncertainty the number is reported with; the overflow it buys is not.
///
/// Returns `(NaN, NaN)` when any draw is not finite, matching what the summary
/// reports for such a parameter, and a `NaN` standard deviation for a single
/// draw, which does not define one. The denominator is `n - 1`: this is the
/// sample standard deviation of the draws, which is what ArviZ's `summary`
/// reports and what the summary table here has always reported.
pub(crate) fn scaled_moments<I, F>(draws: F) -> (f64, f64)
where
    F: Fn() -> I,
    I: Iterator<Item = f64>,
{
    let Some(first) = draws().next() else {
        return (f64::NAN, f64::NAN);
    };
    if draws().any(|x| !x.is_finite()) {
        return (f64::NAN, f64::NAN);
    }

    let mut origin = first;
    let mut scale = draws().map(|x| (x - origin).abs()).fold(0.0, f64::max);
    if !scale.is_finite() {
        origin = 0.0;
        scale = draws().map(|x| x.abs()).fold(0.0, f64::max);
    }
    let divisor = if scale > 0.0 { scale } else { 1.0 };

    let mut sum = 0.0;
    let mut n = 0usize;
    for x in draws() {
        sum += (x - origin) / divisor;
        n += 1;
    }
    let normalized_mean = sum / n as f64;
    let mean = origin + scale * normalized_mean;

    if n < 2 {
        return (mean, f64::NAN);
    }
    let mut sum_sq = 0.0;
    for x in draws() {
        let d = (x - origin) / divisor - normalized_mean;
        sum_sq += d * d;
    }
    (mean, scale * (sum_sq / (n - 1) as f64).sqrt())
}

// ── Internal helpers ────────────────────────────────────────────────

// Diagnostics must not depend on the units of a parameter. Center before
// scaling to preserve small differences around a large common offset; fall
// back to scaling without centering if the finite endpoints span an
// unrepresentable difference. Inputs here have already been shape/finite checked.
fn normalize_chains(chains: &[Vec<f64>]) -> (f64, f64, Vec<Vec<f64>>) {
    let mut origin = chains[0][0];
    let mut scale = chains
        .iter()
        .flatten()
        .map(|x| (x - origin).abs())
        .fold(0.0, f64::max);
    if !scale.is_finite() {
        origin = 0.0;
        scale = chains.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);
    }
    let divisor = if scale > 0.0 { scale } else { 1.0 };
    let normalized = chains
        .iter()
        .map(|chain| chain.iter().map(|x| (x - origin) / divisor).collect())
        .collect();
    (origin, scale, normalized)
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi.min(sorted.len() - 1)] * frac
}

/// Shortest empirical interval containing at least `probability` of the draws.
fn hdi_interval_sorted(sorted: &[f64], probability: f64) -> (f64, f64) {
    if sorted.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return (f64::NAN, f64::NAN);
    }
    let included = ((probability * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    let (start, _) = (0..=sorted.len() - included)
        .map(|start| (start, sorted[start + included - 1] - sorted[start]))
        .min_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("a non-empty sample has at least one HDI candidate");
    (sorted[start], sorted[start + included - 1])
}

/// Rank-normalized, folded split R-hat (Vehtari et al. 2021).
fn r_hat_chains(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    if split.len() < 2 || split.first().is_none_or(|chain| chain.len() < 2) {
        return f64::NAN;
    }

    let rank_r_hat = basic_r_hat(&rank_normalize(&split));
    let mut all: Vec<f64> = split.iter().flatten().copied().collect();
    all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = quantile_sorted(&all, 0.5);
    let folded: Vec<Vec<f64>> = split
        .iter()
        .map(|chain| chain.iter().map(|value| (value - median).abs()).collect())
        .collect();
    let folded_r_hat = basic_r_hat(&rank_normalize(&folded));
    rank_r_hat.max(folded_r_hat)
}

fn basic_r_hat(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;

    let chain_means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance B
    let b = n / (m - 1.0)
        * chain_means
            .iter()
            .map(|&cm| (cm - grand_mean).powi(2))
            .sum::<f64>();

    // Within-chain variance W
    let w = chains
        .iter()
        .map(|c| {
            let cm = mean(c);
            c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>()
        / m;

    if w < 1e-30 {
        return if b < 1e-30 { f64::NAN } else { f64::INFINITY };
    }

    let var_hat = (n - 1.0) / n * w + b / n;
    (var_hat / w).sqrt()
}

/// Bulk ESS using rank-normalized values (Vehtari et al. 2021).
///
/// The chains are split before they are ranked, as Stan and ArviZ do. Ranking
/// first differs for an odd draw count, because the split drops each chain's
/// middle draw and so changes every rank.
fn ess_bulk_chains(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    if !has_enough_split_draws(&split) {
        return f64::NAN;
    }
    ess_split(&rank_normalize(&split))
}

/// Tail ESS: minimum of ESS for the lower and upper tail indicators.
///
/// Both indicators are `x <= q`, as in ArviZ and Stan. The upper one could as
/// well be its complement, which has the same ESS, but `x >= q` is not that
/// complement when a draw sits exactly on the quantile.
fn ess_tail_chains(chains: &[Vec<f64>]) -> f64 {
    let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let below = |q: f64| -> Vec<Vec<f64>> {
        chains
            .iter()
            .map(|c| c.iter().map(|&x| if x <= q { 1.0 } else { 0.0 }).collect())
            .collect()
    };
    let ess_lo = ess_raw(&below(quantile_sorted(&all, 0.05)));
    let ess_hi = ess_raw(&below(quantile_sorted(&all, 0.95)));
    ess_lo.min(ess_hi)
}

/// Rank-normalize: replace values with their normal scores.
fn rank_normalize(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_chains = chains.len();
    let n_per = chains[0].len();
    let total = n_chains * n_per;

    // Collect (value, chain_idx, draw_idx)
    let mut indexed: Vec<(f64, usize, usize)> = Vec::with_capacity(total);
    for (ci, chain) in chains.iter().enumerate() {
        for (di, &v) in chain.iter().enumerate() {
            indexed.push((v, ci, di));
        }
    }
    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (average ties)
    let mut ranks = vec![0.0f64; total];
    let mut i = 0;
    while i < total {
        let mut j = i + 1;
        while j < total && indexed[j].0 == indexed[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        i = j;
    }

    // Blom's normal scores: Φ⁻¹((rank - 3/8) / (N + 1/4)).
    let n_f = total as f64;
    let mut result = vec![vec![0.0; n_per]; n_chains];
    for (idx, &(_, ci, di)) in indexed.iter().enumerate() {
        let p = (ranks[idx] - 0.375) / (n_f + 0.25);
        result[ci][di] = inv_normal_cdf(p);
    }
    result
}

/// ArviZ and Stan need four draws per chain, which leaves two per split half.
fn has_enough_split_draws(split: &[Vec<f64>]) -> bool {
    split.len() >= 2 && split[0].len() >= 2
}

/// ESS of the mean from split chains.
fn ess_raw(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    if !has_enough_split_draws(&split) {
        return f64::NAN;
    }
    ess_split(&split)
}

/// ESS of chains that are already split, by Geyer's initial monotone sequence
/// estimator exactly as Stan and ArviZ implement it.
fn ess_split(split: &[Vec<f64>]) -> f64 {
    let (_, _, split) = normalize_chains(split);
    let m = split.len();
    let n = split[0].len();
    let m_f = m as f64;
    let n_f = n as f64;

    let chain_means: Vec<f64> = split.iter().map(|c| mean(c)).collect();
    let mut acov = LazyAutocovariance::new(&split, &chain_means);
    let acov_0 = acov.at(0);
    let w = acov_0 * n_f / (n_f - 1.0);
    let grand_mean = mean(&chain_means);
    let between = chain_means
        .iter()
        .map(|chain_mean| (chain_mean - grand_mean).powi(2))
        .sum::<f64>()
        / (m_f - 1.0);
    let var_plus = acov_0 + between;
    // Stan returns NaN when every draw is equal; ArviZ returns the draw count.
    // Nothing about such a parameter has an effective sample size.
    if !var_plus.is_finite() || var_plus <= 0.0 {
        return f64::NAN;
    }
    let mut rho = |lag: usize| 1.0 - (w - acov.at(lag)) / var_plus;

    // Geyer's initial positive sequence over pairs (rho_{t+1}, rho_{t+2}),
    // stopping at the first pair whose sum is not positive. A pair that sums
    // below zero is left out; its even term is still kept on its own below.
    let mut rho_hat = vec![0.0; n];
    let mut rho_even = 1.0;
    let mut rho_odd = rho(1);
    rho_hat[0] = rho_even;
    rho_hat[1] = rho_odd;
    let mut t = 1;
    while t + 3 < n && rho_even + rho_odd > 0.0 {
        rho_even = rho(t + 1);
        rho_odd = rho(t + 2);
        if rho_even + rho_odd >= 0.0 {
            rho_hat[t + 1] = rho_even;
            rho_hat[t + 2] = rho_odd;
        }
        t += 2;
    }
    // The retained pairs end at lag `last_even - 1`; `last_even` is the even
    // lag of the pair that ended the sequence. Adding that term on its own
    // when it is positive (Stan's "improved estimate") reduces the variance of
    // the estimate for antithetic chains.
    let last_even = t - 1;
    if rho_even > 0.0 {
        rho_hat[last_even] = rho_even;
    }

    // Geyer's initial monotone sequence.
    let mut t = 1;
    while t + 2 < last_even {
        let previous = rho_hat[t - 1] + rho_hat[t];
        if rho_hat[t + 1] + rho_hat[t + 2] > previous {
            rho_hat[t + 1] = previous / 2.0;
            rho_hat[t + 2] = previous / 2.0;
        }
        t += 2;
    }

    let total_draws = m_f * n_f;
    let tau = -1.0 + 2.0 * rho_hat[..last_even].iter().sum::<f64>() + rho_hat[last_even];
    if rho_hat.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    total_draws / tau.max(1.0 / total_draws.log10())
}

/// Chain-averaged autocovariance, computed only as far as Geyer's sequence
/// asks for it.
///
/// Well-mixed chains end the sequence within a few lags, where the direct sum
/// is cheapest. A slowly mixing chain can run it to lag `n`, where the direct
/// sum costs `O(n²)`; past a cutoff of the order of the transform's own cost
/// every lag comes from [`mean_autocovariance`] instead, so the total stays
/// `O(n log n)`.
struct LazyAutocovariance<'a> {
    chains: &'a [Vec<f64>],
    means: &'a [f64],
    values: Vec<f64>,
    direct_lag_limit: usize,
}

impl<'a> LazyAutocovariance<'a> {
    fn new(chains: &'a [Vec<f64>], means: &'a [f64]) -> Self {
        let n = chains[0].len();
        let log_size = (2 * n).next_power_of_two().trailing_zeros() as usize;
        Self {
            chains,
            means,
            values: Vec::new(),
            direct_lag_limit: 4 * log_size,
        }
    }

    fn at(&mut self, lag: usize) -> f64 {
        while self.values.len() <= lag {
            let next = self.values.len();
            if next >= self.direct_lag_limit {
                self.values = mean_autocovariance(self.chains, self.means);
                break;
            }
            self.values
                .push(direct_autocovariance(self.chains, self.means, next));
        }
        self.values[lag]
    }
}

/// Chain-averaged autocovariance at one lag, with the biased `1/n` estimator
/// Stan and ArviZ use.
fn direct_autocovariance(chains: &[Vec<f64>], means: &[f64], lag: usize) -> f64 {
    let n = chains[0].len();
    let mut gamma = 0.0;
    for (chain, chain_mean) in chains.iter().zip(means) {
        gamma += chain[..n - lag]
            .iter()
            .zip(&chain[lag..])
            .map(|(a, b)| (a - chain_mean) * (b - chain_mean))
            .sum::<f64>();
    }
    gamma / (n * chains.len()) as f64
}

/// Chain-averaged autocovariance at every lag `0..n` through the FFT. Zero
/// padding to at least `2n` makes the circular correlation equal the linear
/// one.
fn mean_autocovariance(chains: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
    let n = chains[0].len();
    let size = (2 * n).next_power_of_two();
    let twiddles = fft_twiddles(size);
    let mut power = vec![0.0; size];
    let mut re = vec![0.0; size];
    let mut im = vec![0.0; size];
    // Two real chains share one complex transform: for z = a + ib,
    // |A_k|² + |B_k|² = (|Z_k|² + |Z_{-k}|²) / 2.
    for (pair, pair_means) in chains.chunks(2).zip(means.chunks(2)) {
        re.fill(0.0);
        im.fill(0.0);
        for (slot, x) in re.iter_mut().zip(&pair[0]) {
            *slot = x - pair_means[0];
        }
        if let (Some(chain), Some(chain_mean)) = (pair.get(1), pair_means.get(1)) {
            for (slot, x) in im.iter_mut().zip(chain) {
                *slot = x - chain_mean;
            }
        }
        fft_in_place(&mut re, &mut im, &twiddles);
        for (k, bin) in power.iter_mut().enumerate() {
            let mirror = (size - k) % size;
            *bin += 0.5
                * (re[k] * re[k]
                    + im[k] * im[k]
                    + re[mirror] * re[mirror]
                    + im[mirror] * im[mirror]);
        }
    }
    // A real, even spectrum is its own forward transform up to the factor
    // `size`, so the inverse transform needs no conjugation.
    re.copy_from_slice(&power);
    im.fill(0.0);
    fft_in_place(&mut re, &mut im, &twiddles);
    let scale = 1.0 / (size as f64 * n as f64 * chains.len() as f64);
    re[..n].iter().map(|value| value * scale).collect()
}

/// `exp(-2πik/size)` for `k < size / 2`, each computed directly so the error
/// does not accumulate the way a recurrence's does.
fn fft_twiddles(size: usize) -> Vec<(f64, f64)> {
    (0..size / 2)
        .map(|k| {
            let (sin, cos) = (-2.0 * std::f64::consts::PI * k as f64 / size as f64).sin_cos();
            (cos, sin)
        })
        .collect()
}

/// Iterative radix-2 forward FFT; `re.len()` must be a power of two.
fn fft_in_place(re: &mut [f64], im: &mut [f64], twiddles: &[(f64, f64)]) {
    let size = re.len();
    let mut j = 0;
    for i in 1..size {
        let mut bit = size >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= size {
        let half = len / 2;
        let stride = size / len;
        for start in (0..size).step_by(len) {
            for k in 0..half {
                let (wr, wi) = twiddles[k * stride];
                let a = start + k;
                let b = a + half;
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut split = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        let mid = chain.len() / 2;
        if mid == 0 {
            continue;
        }
        split.push(chain[..mid].to_vec());
        split.push(chain[chain.len() - mid..].to_vec());
    }
    split
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

/// Approximate inverse standard-normal CDF (Acklam rational approximation).
///
/// This is also used to turn state-space forecast moments into Gaussian
/// pointwise intervals without adding a second implementation.
pub fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmc::TransitionStats;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn inverse_normal_cdf_is_accurate_at_common_interval_levels() {
        let z = inv_normal_cdf(0.975);
        assert!((z - 1.959_963_984_540_054).abs() < 1e-8);
        assert!((inv_normal_cdf(0.025) + z).abs() < 1e-10);
        assert_eq!(inv_normal_cdf(0.5), 0.0);
    }

    #[test]
    fn summaries_and_mean_mcse_preserve_parameter_units() {
        let samples = |offset: f64, scale: f64| {
            (0..4)
                .map(|_| {
                    (0..64)
                        .map(|i| vec![offset + scale * [-3.0, -1.0, 1.0, 3.0][i % 4]])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let report = |offset, scale| {
            compute_diagnostics(&samples(offset, scale), &["x".into()], &[], 0)
                .params
                .remove(0)
        };
        let baseline = report(0.0, 1.0);
        let exact_std = (5.0_f64 * 256.0 / 255.0).sqrt();
        assert!((baseline.std - exact_std).abs() < 1e-14);
        assert!(baseline.mcse_mean.is_finite() && baseline.mcse_mean > 0.0);
        for (offset, scale) in [
            (0.0, 1e-200),
            (0.0, 1e-16),
            (0.0, 1e200),
            (2.0_f64.powi(40), 2.0_f64.powi(-10)),
        ] {
            let scaled = report(offset, scale);
            assert!((scaled.std / scale - exact_std).abs() < 1e-13);
            assert!((scaled.mcse_mean / scale - baseline.mcse_mean).abs() < 1e-13);
            assert!((scaled.ess_bulk - baseline.ess_bulk).abs() < 1e-10);
            assert!((scaled.r_hat - baseline.r_hat).abs() < 1e-13);
        }
    }

    #[test]
    fn finite_large_constant_draws_have_finite_mean_and_zero_std() {
        let report = compute_diagnostics(&vec![vec![vec![1e308]; 16]; 2], &["x".into()], &[], 0);
        assert_eq!(report.params[0].mean, 1e308);
        assert_eq!(report.params[0].std, 0.0);
        assert!(report.params[0].mcse_mean.is_nan());
    }

    /// The converged direction, which the diverged cases below cannot establish:
    /// four chains of genuinely independent draws from one distribution must
    /// score R-hat at the healthy end of the scale.
    ///
    /// The draws are independent normals rather than the deterministic sine
    /// recurrence this test used to feed in. That recurrence is not a sample
    /// from anything, and `< 1.1` is loose enough that the check said little:
    /// R-hat for four independent chains is 1.000 to three decimals, so the
    /// bound is now the 1.01 the samplers are held to.
    #[test]
    fn r_hat_is_near_one_for_independent_chains_of_one_distribution() {
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|seed| {
                let mut rng = ChaCha8Rng::seed_from_u64(7000 + seed);
                (0..1000).map(|_| StandardNormal.sample(&mut rng)).collect()
            })
            .collect();
        let rh = r_hat_chains(&chains);
        assert!(
            rh < 1.01,
            "R-hat should be near 1.0 for independent chains, got {rh}"
        );
    }

    #[test]
    fn test_r_hat_diverged() {
        // Two chains at very different locations
        let chain1: Vec<f64> = (0..500).map(|i| 0.0 + (i as f64 * 0.001)).collect();
        let chain2: Vec<f64> = (0..500).map(|i| 100.0 + (i as f64 * 0.001)).collect();
        let rh = r_hat_chains(&[chain1, chain2]);
        assert!(
            rh > 1.5,
            "R-hat should be large for diverged chains, got {}",
            rh
        );
    }

    #[test]
    fn folded_r_hat_detects_scale_nonconvergence() {
        let narrow: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let wide: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { -10.0 } else { 10.0 })
            .collect();
        let r_hat = r_hat_chains(&[narrow.clone(), narrow, wide.clone(), wide]);
        assert!(
            r_hat > 1.1,
            "folded R-hat failed to detect scale mismatch: {r_hat}"
        );
    }

    /// The independent-draw limit, which the AR(1) and offset cases below do not
    /// pin down: with no autocorrelation the bulk ESS has to come back at the
    /// draw count, not merely above zero.
    ///
    /// `ess > 0` on a deterministic sine wave, which is what this test used to
    /// assert, is satisfied by almost any implementation, including one whose
    /// autocorrelation estimate is off by a constant factor. The measured value
    /// here is 2028.7 against 2000 draws.
    #[test]
    fn ess_matches_the_draw_count_for_independent_draws() {
        const CHAINS: u64 = 4;
        const DRAWS: usize = 500;
        let chains: Vec<Vec<f64>> = (0..CHAINS)
            .map(|seed| {
                let mut rng = ChaCha8Rng::seed_from_u64(8000 + seed);
                (0..DRAWS)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect()
            })
            .collect();
        let total = (CHAINS as usize * DRAWS) as f64;
        let ess = ess_bulk_chains(&chains);
        assert!(
            (ess - total).abs() / total < 0.05,
            "bulk ESS {ess} of independent draws differs too much from the draw count {total}"
        );
    }

    #[test]
    fn ess_tracks_ar1_closed_form() {
        const PHI: f64 = 0.5;
        const CHAINS: usize = 4;
        const DRAWS: usize = 1000;
        let mut chains = Vec::with_capacity(CHAINS);
        for seed in 0..CHAINS {
            let mut rng = ChaCha8Rng::seed_from_u64(100 + seed as u64);
            let mut state = 0.0;
            let mut chain = Vec::with_capacity(DRAWS);
            for _ in 0..DRAWS {
                let innovation: f64 = StandardNormal.sample(&mut rng);
                state = PHI * state + innovation;
                chain.push(state);
            }
            chains.push(chain);
        }

        let actual = ess_raw(&chains);
        let total = (CHAINS * DRAWS) as f64;
        let expected = total * (1.0 - PHI) / (1.0 + PHI);
        assert!(
            (actual - expected).abs() / expected < 0.30,
            "AR(1) ESS {actual} differs too much from closed-form {expected}"
        );
    }

    #[test]
    fn ess_accounts_for_between_chain_offsets() {
        let base: Vec<f64> = (0..1000).map(|i| ((i as f64) * 0.173).sin()).collect();
        let chains: Vec<Vec<f64>> = [-15.0, -5.0, 5.0, 15.0]
            .iter()
            .map(|offset| base.iter().map(|value| value + offset).collect())
            .collect();

        let ess = ess_raw(&chains);
        assert!(
            ess < 100.0,
            "ESS ignored persistent between-chain offsets: {ess}"
        );
    }

    /// SplitMix64 uniforms summed into Irwin-Hall normals. Every step is exact
    /// or correctly rounded, so a Python port regenerates these arrays bit for
    /// bit; that is how the ArviZ references below were produced.
    struct ReferenceStream(u64);

    impl ReferenceStream {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            let mut acc = 0.0;
            for _ in 0..12 {
                acc += self.uniform();
            }
            acc - 6.0
        }
    }

    fn reference_ar1(
        seed: u64,
        chains: usize,
        draws: usize,
        rho: f64,
        offsets: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut stream = ReferenceStream(seed);
        let innovation = (1.0 - rho * rho).sqrt();
        (0..chains)
            .map(|chain| {
                let mut x = stream.normal();
                (0..draws)
                    .map(|_| {
                        x = rho * x + innovation * stream.normal();
                        x + offsets.get(chain).copied().unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect()
    }

    struct ArvizReference {
        name: &'static str,
        chains: Vec<Vec<f64>>,
        ess_bulk: f64,
        ess_tail: f64,
        r_hat: f64,
        mcse_mean: f64,
        ess_mean: f64,
    }

    /// Values from ArviZ 0.23.4 (`_ess_bulk`, `_ess_tail`, `_rhat_rank`,
    /// `_mcse_mean`, `_ess_mean`) on the same arrays. Both short cases hit the
    /// `N log10 N` ceiling, which pins the split draw count; the antithetic
    /// case is where the final even-lag term Geyer's sequence adds matters most.
    fn arviz_references() -> Vec<ArvizReference> {
        let floor_half = |chains: Vec<Vec<f64>>| {
            chains
                .into_iter()
                .map(|chain| chain.into_iter().map(|x| (x * 2.0).floor() / 2.0).collect())
                .collect()
        };
        vec![
            ArvizReference {
                name: "iid",
                chains: reference_ar1(1, 4, 200, 0.0, &[]),
                ess_bulk: 900.843_413_070_252_4,
                ess_tail: 772.755_599_201_520_7,
                r_hat: 0.999_338_535_778_982_8,
                mcse_mean: 0.032_446_956_788_381_304,
                ess_mean: 902.431_198_968_420_6,
            },
            ArvizReference {
                name: "ar1_0.9",
                chains: reference_ar1(2, 4, 500, 0.9, &[]),
                ess_bulk: 104.382_333_518_397_56,
                ess_tail: 182.817_875_209_919,
                r_hat: 1.035_508_539_822_694,
                mcse_mean: 0.094_733_472_621_408_65,
                ess_mean: 104.432_953_482_784_84,
            },
            // Mixes slowly enough that the autocovariance comes from the FFT.
            ArvizReference {
                name: "ar1_0.99",
                chains: reference_ar1(9, 4, 500, 0.99, &[]),
                ess_bulk: 10.446_805_996_995_531,
                ess_tail: 20.230_457_184_539_848,
                r_hat: 1.331_419_289_025_579_4,
                mcse_mean: 0.271_148_959_894_050_9,
                ess_mean: 9.479_811_464_859_088,
            },
            ArvizReference {
                name: "antithetic",
                chains: reference_ar1(3, 4, 300, -0.3, &[]),
                ess_bulk: 2_748.516_392_480_128,
                ess_tail: 1_267.719_343_613_817_3,
                r_hat: 1.000_512_372_543_435_5,
                mcse_mean: 0.019_494_792_461_899_585,
                ess_mean: 2_739.842_654_991_770_4,
            },
            ArvizReference {
                name: "offset",
                chains: reference_ar1(4, 4, 200, 0.3, &[0.0, 0.5, -0.5, 1.0]),
                ess_bulk: 17.238_108_979_497_014,
                ess_tail: 255.284_430_352_126_72,
                r_hat: 1.173_098_755_696_653_5,
                mcse_mean: 0.280_601_358_552_275_75,
                ess_mean: 16.870_940_297_474_128,
            },
            ArvizReference {
                name: "short_odd",
                chains: reference_ar1(5, 3, 9, 0.2, &[]),
                ess_bulk: 33.125_069_801_078_54,
                ess_tail: 33.125_069_801_078_54,
                r_hat: 1.107_470_199_310_643,
                mcse_mean: 0.177_186_480_423_544_15,
                ess_mean: 33.125_069_801_078_54,
            },
            ArvizReference {
                name: "short_min",
                chains: reference_ar1(6, 2, 5, 0.0, &[]),
                ess_bulk: 7.224_719_895_935_548,
                ess_tail: 7.224_719_895_935_548,
                r_hat: 0.952_444_752_958_862,
                mcse_mean: 0.263_713_943_473_540_6,
                ess_mean: 7.224_719_895_935_548,
            },
            ArvizReference {
                name: "short_ar",
                chains: reference_ar1(8, 4, 21, 0.6, &[]),
                ess_bulk: 37.208_185_748_418_195,
                ess_tail: 51.421_973_228_413_82,
                r_hat: 1.054_814_668_125_427,
                mcse_mean: 0.178_990_320_460_839_85,
                ess_mean: 36.953_924_554_091_89,
            },
            ArvizReference {
                name: "ties",
                chains: floor_half(reference_ar1(7, 4, 100, 0.5, &[])),
                ess_bulk: 119.228_701_387_085_92,
                ess_tail: 242.216_179_110_383_56,
                r_hat: 1.022_407_235_815_689,
                mcse_mean: 0.091_672_763_899_559_7,
                ess_mean: 119.089_171_579_429_4,
            },
        ]
    }

    #[test]
    fn reference_stream_reproduces_the_python_arrays() {
        let iid = reference_ar1(1, 4, 200, 0.0, &[]);
        assert_eq!(
            iid[0][..3],
            [-0.6189042598785681, -0.5907667571367323, 1.4329694955301964]
        );
        let offset = reference_ar1(4, 4, 200, 0.3, &[0.0, 0.5, -0.5, 1.0]);
        assert_eq!(
            offset[0][..3],
            [0.05139799030537601, 1.0157946443861814, 0.9732998827049826]
        );
    }

    /// The mean ESS involves no normal quantile, so it has to agree with ArviZ
    /// to rounding. The rank-based statistics go through the Acklam inverse
    /// normal CDF, whose relative error is about 1e-9, where ArviZ calls SciPy.
    #[test]
    fn diagnostics_match_arviz_references() {
        for case in arviz_references() {
            let rel = |actual: f64, expected: f64| (actual - expected).abs() / expected.abs();
            let ess_mean = ess_raw(&case.chains);
            assert!(
                rel(ess_mean, case.ess_mean) < 1e-10,
                "{}: mean ESS {ess_mean} vs ArviZ {}",
                case.name,
                case.ess_mean
            );

            let samples: Vec<Vec<Vec<f64>>> = case
                .chains
                .iter()
                .map(|chain| chain.iter().map(|&x| vec![x]).collect())
                .collect();
            let report = compute_diagnostics(&samples, &["x".into()], &[], 0);
            let p = &report.params[0];
            for (label, actual, expected) in [
                ("ess_bulk", p.ess_bulk, case.ess_bulk),
                ("ess_tail", p.ess_tail, case.ess_tail),
                ("r_hat", p.r_hat, case.r_hat),
                ("mcse_mean", p.mcse_mean, case.mcse_mean),
            ] {
                assert!(
                    rel(actual, expected) < 1e-7,
                    "{}: {label} {actual} vs ArviZ {expected}",
                    case.name
                );
            }
        }
    }

    /// ArviZ returns the draw count as the ESS of a constant array; Stan and the
    /// R `posterior` package return NaN, which is what a parameter that never
    /// moved should report, since no draw count describes it.
    #[test]
    fn constant_draws_have_no_effective_sample_size() {
        let chains = vec![vec![3.0; 100]; 4];
        assert!(ess_raw(&chains).is_nan());
        assert!(ess_bulk_chains(&chains).is_nan());
        assert!(ess_tail_chains(&chains).is_nan());
        assert!(r_hat_chains(&chains).is_nan());
    }

    /// Chains that each sit still at different values have no within-chain
    /// variance but a well-defined `var_plus`, so ArviZ and Stan report a tiny
    /// ESS rather than none.
    #[test]
    fn stuck_chains_report_a_small_ess_rather_than_none() {
        let chains: Vec<Vec<f64>> = [0.0, 1.0, 2.0, 3.0]
            .iter()
            .map(|&value| vec![value; 100])
            .collect();
        let ess = ess_raw(&chains);
        // ArviZ 0.23.4 gives 100 / 23 for these chains.
        assert!((ess - 100.0 / 23.0).abs() < 1e-12, "{ess}");
    }

    #[test]
    fn fft_autocovariance_matches_the_direct_sum() {
        let direct = |chains: &[Vec<f64>], means: &[f64]| -> Vec<f64> {
            let n = chains[0].len();
            (0..n)
                .map(|lag| {
                    let mut gamma = 0.0;
                    for (chain, chain_mean) in chains.iter().zip(means) {
                        for t in 0..n - lag {
                            gamma += (chain[t] - chain_mean) * (chain[t + lag] - chain_mean);
                        }
                    }
                    gamma / (n * chains.len()) as f64
                })
                .collect()
        };
        for (seed, chains, draws, rho) in [
            (11, 8, 500, 0.95),
            (12, 3, 37, -0.4),
            (13, 1, 2, 0.0),
            (14, 5, 1024, 0.5),
            (15, 2, 1025, 0.999),
        ] {
            let chains = reference_ar1(seed, chains, draws, rho, &[]);
            let means: Vec<f64> = chains.iter().map(|chain| mean(chain)).collect();
            let fft = mean_autocovariance(&chains, &means);
            let expected = direct(&chains, &means);
            assert_eq!(fft.len(), expected.len());
            let mut lazy = LazyAutocovariance::new(&chains, &means);
            for (lag, (a, b)) in fft.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * expected[0],
                    "seed {seed} lag {lag}: FFT {a} vs direct {b}"
                );
                let c = lazy.at(lag);
                assert!(
                    (c - b).abs() <= 1e-10 * expected[0],
                    "seed {seed} lag {lag}: lazy {c} vs direct {b}"
                );
            }
        }
    }

    #[test]
    fn reported_interval_is_a_highest_density_interval() {
        let mut draws = vec![0.0; 94];
        draws.extend([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let (lower, upper) = hdi_interval_sorted(&draws, 0.94);
        assert_eq!((lower, upper), (0.0, 0.0));
        assert_ne!(upper, quantile_sorted(&draws, 0.97));
    }

    #[test]
    fn diagnostics_warn_at_modern_r_hat_threshold() {
        let report = DiagnosticsReport {
            params: vec![ParamDiagnostics {
                name: "theta".into(),
                mean: 0.0,
                std: 1.0,
                hdi_3: -1.0,
                hdi_97: 1.0,
                ess_bulk: 1000.0,
                ess_tail: 1000.0,
                r_hat: 1.02,
                mcse_mean: 0.01,
            }],
            num_chains: 4,
            num_draws: 1000,
            accept_rates: vec![0.8; 4],
            divergences: 0,
        };

        assert!(report.to_table().contains("R-hat values > 1.01"));
    }

    #[test]
    fn test_transition_diagnostics_aggregate() {
        let transitions = vec![
            vec![
                TransitionStats {
                    is_warmup: true,
                    accepted: true,
                    accept_prob: 0.9,
                    energy_error: 0.1,
                    divergent: true,
                    step_size: 1.0,
                    num_leapfrog_steps: 5,
                    tree_depth: Some(2),
                },
                TransitionStats {
                    is_warmup: false,
                    accepted: false,
                    accept_prob: 0.7,
                    energy_error: -0.4,
                    divergent: true,
                    step_size: 1.0,
                    num_leapfrog_steps: 7,
                    tree_depth: Some(3),
                },
            ],
            vec![TransitionStats {
                is_warmup: false,
                accepted: true,
                accept_prob: 0.8,
                energy_error: 0.2,
                divergent: false,
                step_size: 0.5,
                num_leapfrog_steps: 6,
                tree_depth: None,
            }],
        ];

        let report = compute_transition_diagnostics(&transitions);
        assert_eq!(report.total_transitions, 3);
        assert_eq!(report.total_warmup_transitions, 1);
        assert_eq!(report.total_draw_transitions, 2);
        // Warmup telemetry remains in the input, but user-facing diagnostics
        // describe the returned posterior draws only.
        assert_eq!(report.total_divergences, 1);
        assert_eq!(report.total_leapfrog_steps, 18);
        assert_eq!(report.chains.len(), 2);
        assert_eq!(report.chains[0].divergences, 1);
        assert_eq!(report.chains[0].total_leapfrog_steps, 12);
        assert_eq!(report.chains[1].total_leapfrog_steps, 6);
        assert_eq!(report.chains[1].max_tree_depth, None);
        assert_eq!(report.mean_accept_prob, 0.75);
        assert!(report.max_abs_energy_error >= 0.4);
    }

    #[test]
    fn transition_diagnostics_keep_warmup_work_out_of_posterior_rates() {
        let transitions = vec![vec![TransitionStats {
            is_warmup: true,
            accepted: true,
            accept_prob: 0.9,
            energy_error: 0.25,
            divergent: true,
            step_size: 0.5,
            num_leapfrog_steps: 9,
            tree_depth: Some(4),
        }]];

        let report = compute_transition_diagnostics(&transitions);
        assert_eq!(report.total_divergences, 0);
        assert_eq!(report.mean_accept_prob, 0.0);
        assert_eq!(report.total_leapfrog_steps, 9);
        assert_eq!(report.mean_energy_error, 0.25);
        assert_eq!(report.chains[0].accepted_transitions, 0);
        assert_eq!(report.chains[0].mean_step_size, 0.5);
        assert_eq!(report.chains[0].max_tree_depth, Some(4));
    }

    // ---- table layout ---------------------------------------------------

    /// Start/end char offsets of every whitespace-delimited field in `line`.
    fn row_fields(line: &str) -> Vec<(usize, usize)> {
        let mut fields = Vec::new();
        let mut start: Option<usize> = None;
        for (index, ch) in line.chars().enumerate() {
            match (ch.is_whitespace(), start) {
                (false, None) => start = Some(index),
                (true, Some(begin)) => {
                    fields.push((begin, index));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = start {
            fields.push((begin, line.chars().count()));
        }
        fields
    }

    /// Assert that every row between the two rules occupies exactly the same
    /// columns as the header row, and return the table width.
    ///
    /// The first column is left aligned, so its fields all start at 0; every
    /// other column is right aligned, so its fields all end together. Widths
    /// are measured in chars, because the rule is drawn with a 3-byte
    /// character.
    fn assert_columns_aligned(table: &str) -> usize {
        let lines: Vec<&str> = table.lines().collect();
        let top = lines
            .iter()
            .position(|line| line.starts_with('─'))
            .expect("table has an opening rule");
        let bottom = lines
            .iter()
            .rposition(|line| line.starts_with('─'))
            .expect("table has a closing rule");
        assert!(bottom > top, "expected two distinct rules");

        let width = lines[top].chars().count();
        assert_eq!(
            lines[bottom].chars().count(),
            width,
            "the two rules disagree about the table width"
        );

        let header = lines[top - 1];
        let expected = row_fields(header);
        for row in std::iter::once(header).chain(lines[top + 1..bottom].iter().copied()) {
            assert_eq!(
                row.chars().count(),
                width,
                "row width {} disagrees with rule width {width}:\n{row}",
                row.chars().count()
            );
            let fields = row_fields(row);
            assert_eq!(
                fields.len(),
                expected.len(),
                "row has the wrong number of columns:\n{row}"
            );
            assert_eq!(fields[0].0, 0, "first column is not flush left:\n{row}");
            for (index, (field, reference)) in fields.iter().zip(&expected).enumerate().skip(1) {
                assert_eq!(
                    field.1, reference.1,
                    "column {index} ends at {} in\n{row}\nbut at {} in\n{header}",
                    field.1, reference.1
                );
            }
        }
        width
    }

    fn report_for_names(names: &[&str]) -> DiagnosticsReport {
        DiagnosticsReport {
            params: names
                .iter()
                .map(|name| ParamDiagnostics {
                    name: (*name).to_string(),
                    mean: 0.0603,
                    std: 0.0225,
                    hdi_3: 0.0267,
                    hdi_97: 0.1052,
                    ess_bulk: 1580.0,
                    ess_tail: 1338.0,
                    r_hat: 1.0001,
                    mcse_mean: 0.001796,
                })
                .collect(),
            num_chains: 4,
            num_draws: 500,
            accept_rates: vec![0.9; 4],
            divergences: 0,
        }
    }

    #[test]
    fn short_name_tables_keep_their_historical_layout() {
        let table = report_for_names(&["mu", "sigma"]).to_table();
        let width = assert_columns_aligned(&table);
        // 12-wide name column, eight numeric columns, eight separating spaces.
        // (The rules used to be hard-coded at 96, two columns too wide.)
        assert_eq!(width, 94);
        let mut lines = table
            .lines()
            .skip_while(|line| !line.starts_with("Parameter"));
        // Header and first data row, character for character: this pins the
        // displayed values and their precision as well as the layout.
        assert_eq!(
            lines.next(),
            Some(
                "Parameter        mean      std     hdi_3%    hdi_97%   ess_bulk   ess_tail    r_hat  mcse_mean"
            )
        );
        assert!(lines.next().is_some_and(|line| line.starts_with('─')));
        assert_eq!(
            lines.next(),
            Some(
                "mu             0.0603   0.0225     0.0267     0.1052       1580       1338   1.0001   0.001796"
            )
        );
    }

    #[test]
    fn long_parameter_names_do_not_shift_the_numeric_columns() {
        // Names this library's own BayesianLocalLevel model emits.
        let table = report_for_names(&[
            "process_variance",
            "observation_variance",
            "terminal_level",
            "mu",
        ])
        .to_table();
        let width = assert_columns_aligned(&table);
        assert_eq!(width, 94 - 12 + "observation_variance".len());
        for name in ["process_variance", "observation_variance", "terminal_level"] {
            assert!(table.contains(name), "{name} missing from table");
        }
    }

    #[test]
    fn wide_numeric_values_do_not_shift_later_columns() {
        let mut report = report_for_names(&["mu", "sigma"]);
        report.params[0].mean = -123_456.789;
        report.params[0].mcse_mean = -98_765.432_1;
        report.params[1].ess_bulk = 12_345_678.0;
        let table = report.to_table();
        let width = assert_columns_aligned(&table);
        assert!(
            width > 94,
            "columns must grow for oversized values, got {width}"
        );
        assert!(
            table.contains("-123456.7890"),
            "value was altered:\n{table}"
        );
    }

    #[test]
    fn pathological_names_are_abbreviated_but_stay_aligned() {
        let long = "z".repeat(200);
        let table = report_for_names(&[&long, "mu"]).to_table();
        let width = assert_columns_aligned(&table);
        assert_eq!(width, 94 - 12 + NAME_COLUMN_MAX_WIDTH);
        assert!(table.contains('…'), "expected an ellipsis in\n{table}");
        assert!(!table.contains(&long), "200-char name was not abbreviated");
    }

    #[test]
    fn abbreviation_never_makes_two_parameters_print_identically() {
        // These names share a 40-char head and a 16-char tail, so abbreviating
        // them would render them identically; the table stays wide instead.
        let head = "p".repeat(40);
        let tail = "s".repeat(16);
        let first = format!("{head}A{tail}");
        let second = format!("{head}B{tail}");
        let table = report_for_names(&[&first, &second]).to_table();
        let width = assert_columns_aligned(&table);
        assert_eq!(width, 94 - 12 + first.chars().count());
        assert!(table.contains(&first) && table.contains(&second));
        assert!(!table.contains('…'));
    }

    #[test]
    fn columns_align_when_a_parameter_is_unavailable() {
        let mut report = report_for_names(&["mu", "observation_variance"]);
        report.params[1] = unavailable_parameter("observation_variance".to_string());
        let table = report.to_table();
        assert_columns_aligned(&table);
        assert!(table.contains("NaN"));
    }

    #[test]
    fn abbreviation_accounts_for_the_padding_the_column_adds() {
        // The first name abbreviates to the second one plus a trailing space:
        // distinct as strings, identical once the column pads them out. The
        // uniqueness check has to compare labels as they will be printed.
        let first = format!("{}MIDDLE{} ", "a".repeat(31), "b".repeat(15));
        let second = format!("{}…{}", "a".repeat(31), "b".repeat(15));
        assert_eq!(
            abbreviate(&first, NAME_COLUMN_MAX_WIDTH).trim_end_matches(' '),
            second
        );

        let table = report_for_names(&[&first, &second]).to_table();
        let width = assert_columns_aligned(&table);
        assert_eq!(width, 94 - 12 + first.chars().count());
        let body: Vec<&str> = table.lines().filter(|line| line.starts_with('a')).collect();
        assert_eq!(body.len(), 2, "expected two parameter rows in\n{table}");
        assert_ne!(body[0], body[1], "two parameters printed identically");
    }

    #[test]
    fn transition_table_columns_grow_with_their_contents() {
        let transitions = vec![
            vec![TransitionStats {
                is_warmup: false,
                accepted: true,
                accept_prob: 0.9,
                energy_error: 0.1,
                divergent: false,
                step_size: 1.0,
                num_leapfrog_steps: 5,
                tree_depth: Some(2),
            }],
            vec![TransitionStats {
                is_warmup: false,
                accepted: true,
                accept_prob: 0.8,
                // Renders as 13 chars in a 12-wide column ...
                energy_error: -1_234_567.89,
                divergent: false,
                step_size: 0.5,
                // ... and 12 chars in a 10-wide one.
                num_leapfrog_steps: 123_456_789_012,
                tree_depth: None,
            }],
        ];
        let table = compute_transition_diagnostics(&transitions).to_table();
        let width = assert_columns_aligned(&table);
        assert!(
            table.contains("-1234567.8900"),
            "value was altered:\n{table}"
        );
        assert!(
            table.contains("123456789012"),
            "value was altered:\n{table}"
        );
        // 96 is the sum of the historical minimum widths, which the rule used
        // to overstate as 100; the two oversized cells add 1 and 2 columns.
        assert_eq!(width, 96 + 1 + 2);
    }

    #[test]
    fn transition_table_rules_match_the_rendered_width() {
        let transitions = vec![vec![TransitionStats {
            is_warmup: false,
            accepted: true,
            accept_prob: 0.9,
            energy_error: 0.1,
            divergent: false,
            step_size: 1.0,
            num_leapfrog_steps: 5,
            tree_depth: Some(2),
        }]];
        let table = compute_transition_diagnostics(&transitions).to_table();
        assert_eq!(assert_columns_aligned(&table), 96);
    }
}
