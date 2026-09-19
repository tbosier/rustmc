use crate::hmc::TransitionStats;
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

    let mut params = Vec::with_capacity(n_params);

    for pidx in 0..n_params {
        // Extract per-chain traces for this parameter
        let chains: Vec<Vec<f64>> = (0..n_chains)
            .map(|c| samples[c].iter().map(|draw| draw[pidx]).collect())
            .collect();

        if chains.iter().flatten().any(|value| !value.is_finite()) {
            params.push(unavailable_parameter(param_names[pidx].clone()));
            continue;
        }
        let (mean, std) = scaled_moments(|| chains.iter().flatten().copied());
        let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
        all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (hdi_3, hdi_97) = hdi_interval_sorted(&all, 0.94);
        let ess_bulk = ess_bulk_chains(&chains);
        let ess_tail = ess_tail_chains(&chains);
        let r_hat = r_hat_chains(&chains);
        let ess_mean = ess_raw(&chains);
        let mcse_mean = if ess_mean > 0.0 {
            std / ess_mean.sqrt()
        } else {
            f64::NAN
        };

        params.push(ParamDiagnostics {
            name: param_names[pidx].clone(),
            mean,
            std,
            hdi_3,
            hdi_97,
            ess_bulk,
            ess_tail,
            r_hat,
            mcse_mean,
        });
    }

    DiagnosticsReport {
        params,
        num_chains: n_chains,
        num_draws: n_draws,
        accept_rates: accept_rates.to_vec(),
        divergences,
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
fn ess_bulk_chains(chains: &[Vec<f64>]) -> f64 {
    let ranked = rank_normalize(chains);
    ess_raw(&ranked)
}

/// Tail ESS: minimum of ESS for the lower and upper tail indicators.
fn ess_tail_chains(chains: &[Vec<f64>]) -> f64 {
    let all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    let q05 = {
        let mut s = all.clone();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        quantile_sorted(&s, 0.05)
    };
    let q95 = {
        let mut s = all;
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        quantile_sorted(&s, 0.95)
    };

    let lower: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| {
            c.iter()
                .map(|&x| if x <= q05 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let upper: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| {
            c.iter()
                .map(|&x| if x >= q95 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    let ess_lo = ess_raw(&lower);
    let ess_hi = ess_raw(&upper);
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

    // Normal scores: Φ⁻¹((rank - 3/8) / (N - 1/4))
    let n_f = total as f64;
    let mut result = vec![vec![0.0; n_per]; n_chains];
    for (idx, &(_, ci, di)) in indexed.iter().enumerate() {
        let p = (ranks[idx] - 0.375) / (n_f + 0.25);
        result[ci][di] = inv_normal_cdf(p);
    }
    result
}

/// ESS from split chains using autocorrelation (Geyer's initial monotone sequence).
fn ess_raw(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    if split.len() < 2 || split.first().is_none_or(|chain| chain.len() < 3) {
        return f64::NAN;
    }
    let (_, _, split) = normalize_chains(&split);
    let m = split.len();
    let n = split[0].len();

    let chain_means: Vec<f64> = split.iter().map(|c| mean(c)).collect();
    let m_f = m as f64;
    let n_f = n as f64;

    let w: f64 = split
        .iter()
        .map(|c| {
            let cm = mean(c);
            c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n_f - 1.0)
        })
        .sum::<f64>()
        / m_f;

    if w <= 0.0 {
        return f64::NAN;
    }

    let b = n_f / (m_f - 1.0)
        * chain_means
            .iter()
            .map(|chain_mean| (chain_mean - mean(&chain_means)).powi(2))
            .sum::<f64>();
    let var_plus = (n_f - 1.0) / n_f * w + b / n_f;
    if !var_plus.is_finite() || var_plus <= 0.0 {
        return f64::NAN;
    }

    // Estimate autocorrelations with V-hat-plus in the denominator. The
    // autocovariance uses the biased (1 / n) estimator used by Stan/ArviZ.
    let rho_at = |lag: usize| {
        let mut gamma = 0.0f64;
        for (ci, chain) in split.iter().enumerate() {
            let cm = chain_means[ci];
            let valid = n - lag;
            for t in 0..valid {
                gamma += (chain[t] - cm) * (chain[t + lag] - cm);
            }
        }
        gamma /= m_f * n_f;
        1.0 - (w - gamma) / var_plus
    };

    // Geyer's initial positive sequence, followed by the initial monotone
    // sequence. The first pair includes rho_0 = 1.
    let mut pair_sums = Vec::new();
    let mut lag = 1;
    while lag < n {
        let rho_even = if lag == 1 { 1.0 } else { rho_at(lag - 1) };
        let rho_odd = rho_at(lag);
        let mut pair_sum = rho_even + rho_odd;
        if !pair_sum.is_finite() || pair_sum < 0.0 {
            break;
        }
        if let Some(previous) = pair_sums.last() {
            pair_sum = pair_sum.min(*previous);
        }
        pair_sums.push(pair_sum);
        lag += 2;
    }

    let total_draws = m_f * n_f;
    let tau = (-1.0 + 2.0 * pair_sums.iter().sum::<f64>()).max(1.0 / total_draws.log10());
    total_draws / tau
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
