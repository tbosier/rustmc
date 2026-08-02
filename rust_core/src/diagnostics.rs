use crate::hmc::TransitionStats;

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

impl DiagnosticsReport {
    /// Render the diagnostics as a formatted table string.
    pub fn to_table(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{} chains × {} draws per chain",
            self.num_chains, self.num_draws
        ));
        lines.push(String::new());
        lines.push(format!(
            "{:<12} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "Parameter",
            "mean",
            "std",
            "hdi_3%",
            "hdi_97%",
            "ess_bulk",
            "ess_tail",
            "r_hat",
            "mcse_mean"
        ));
        lines.push("─".repeat(96));

        for p in &self.params {
            let ess_bulk_s = if p.ess_bulk.is_finite() {
                format!("{:.0}", p.ess_bulk)
            } else {
                "NaN".to_string()
            };
            let ess_tail_s = if p.ess_tail.is_finite() {
                format!("{:.0}", p.ess_tail)
            } else {
                "NaN".to_string()
            };
            lines.push(format!(
                "{:<12} {:>8.4} {:>8.4} {:>10.4} {:>10.4} {:>10} {:>10} {:>8.4} {:>10.6}",
                p.name,
                p.mean,
                p.std,
                p.hdi_3,
                p.hdi_97,
                ess_bulk_s,
                ess_tail_s,
                p.r_hat,
                p.mcse_mean,
            ));
        }

        lines.push("─".repeat(96));

        let avg_accept: f64 = if self.accept_rates.is_empty() {
            0.0
        } else {
            self.accept_rates.iter().sum::<f64>() / self.accept_rates.len() as f64
        };
        lines.push(format!(
            "Mean accept rate: {:.2}  │  Divergences: {}",
            avg_accept, self.divergences
        ));

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
        if self.divergences > 0 {
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
        lines.push(format!(
            "{:<6} {:>8} {:>8} {:>10} {:>10} {:>12} {:>12} {:>12} {:>10}",
            "chain", "trans", "warmup", "div", "acc", "mean_acc", "mean_dH", "max|dH|", "leapfrogs"
        ));
        lines.push("─".repeat(100));

        for chain in &self.chains {
            lines.push(format!(
                "{:<6} {:>8} {:>8} {:>10} {:>10} {:>12.4} {:>12.4} {:>12.4} {:>10}",
                chain.chain_index,
                chain.num_transitions,
                chain.num_warmup_transitions,
                chain.divergences,
                chain.accepted_transitions,
                chain.mean_accept_prob,
                chain.mean_energy_error,
                chain.max_abs_energy_error,
                chain.total_leapfrog_steps
            ));
        }

        lines.push("─".repeat(100));
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

    let mut params = Vec::with_capacity(n_params);

    for pidx in 0..n_params {
        // Extract per-chain traces for this parameter
        let chains: Vec<Vec<f64>> = (0..n_chains)
            .map(|c| samples[c].iter().map(|draw| draw[pidx]).collect())
            .collect();

        let mean = chain_mean_all(&chains);
        let std = chain_std_all(&chains, mean);
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
            total_leapfrog_steps,
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

// ── Internal helpers ────────────────────────────────────────────────

fn chain_mean_all(chains: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for c in chains {
        for &v in c {
            sum += v;
            n += 1;
        }
    }
    sum / n as f64
}

fn chain_std_all(chains: &[Vec<f64>], mean: f64) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for c in chains {
        for &v in c {
            let d = v - mean;
            sum_sq += d * d;
            n += 1;
        }
    }
    (sum_sq / (n - 1) as f64).sqrt()
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
        let mut j = i;
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

    if w < 1e-30 {
        return f64::NAN;
    }

    let b = n_f / (m_f - 1.0)
        * chain_means
            .iter()
            .map(|chain_mean| (chain_mean - mean(&chain_means)).powi(2))
            .sum::<f64>();
    let var_plus = (n_f - 1.0) / n_f * w + b / n_f;
    if !var_plus.is_finite() || var_plus < 1e-30 {
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
    fn test_r_hat_converged() {
        // Four chains sampling from the same distribution should have R-hat ≈ 1.0
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|seed| {
                let mut rng = seed as f64;
                (0..1000)
                    .map(|i| {
                        rng = (rng * 1.1 + 0.3).sin() * 10.0;
                        rng + (i as f64 * 0.001)
                    })
                    .collect()
            })
            .collect();
        let rh = r_hat_chains(&chains);
        assert!(
            rh < 1.1,
            "R-hat should be near 1.0 for converged chains, got {}",
            rh
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

    #[test]
    fn test_ess_positive() {
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|seed| {
                (0..500)
                    .map(|i| ((seed * 1000 + i) as f64 * 0.1).sin() * 2.0)
                    .collect()
            })
            .collect();
        let ess = ess_bulk_chains(&chains);
        assert!(ess > 0.0, "ESS should be positive, got {}", ess);
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
}
