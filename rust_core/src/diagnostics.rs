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
            "Parameter", "mean", "std", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat", "mcse_mean"
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
                p.name, p.mean, p.std, p.hdi_3, p.hdi_97, ess_bulk_s, ess_tail_s, p.r_hat,
                p.mcse_mean,
            ));
        }

        lines.push("─".repeat(96));

        let avg_accept: f64 =
            self.accept_rates.iter().sum::<f64>() / self.accept_rates.len() as f64;
        lines.push(format!(
            "Mean accept rate: {:.2}  │  Divergences: {}",
            avg_accept, self.divergences
        ));

        let any_bad_rhat = self.params.iter().any(|p| p.r_hat > 1.05 || !p.r_hat.is_finite());
        let any_low_ess = self.params.iter().any(|p| p.ess_bulk < 400.0 || p.ess_tail < 400.0);

        if any_bad_rhat {
            lines.push(
                "⚠  Some R-hat values > 1.05 — chains may not have converged.".to_string(),
            );
        }
        if any_low_ess {
            lines.push(
                "⚠  Some ESS values < 400 — consider increasing draws or tuning.".to_string(),
            );
        }
        if self.divergences > 0 {
            lines.push(format!(
                "⚠  {} divergent transitions — results may be unreliable.",
                self.divergences
            ));
        }

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
        let hdi_3 = quantile_sorted(&all, 0.03);
        let hdi_97 = quantile_sorted(&all, 0.97);
        let ess_bulk = ess_bulk_chains(&chains);
        let ess_tail = ess_tail_chains(&chains);
        let r_hat = r_hat_chains(&chains);
        let mcse_mean = if ess_bulk > 0.0 {
            std / ess_bulk.sqrt()
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

/// Split R-hat: split each chain in half, treat as 2M chains, compute R-hat.
fn r_hat_chains(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    let m = split.len() as f64;
    let n = split[0].len() as f64;

    let chain_means: Vec<f64> = split.iter().map(|c| mean(c)).collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance B
    let b = n / (m - 1.0)
        * chain_means
            .iter()
            .map(|&cm| (cm - grand_mean).powi(2))
            .sum::<f64>();

    // Within-chain variance W
    let w = split
        .iter()
        .map(|c| {
            let cm = mean(c);
            c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>()
        / m;

    if w < 1e-30 {
        return f64::NAN;
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
        .map(|c| c.iter().map(|&x| if x <= q05 { 1.0 } else { 0.0 }).collect())
        .collect();
    let upper: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|&x| if x >= q95 { 1.0 } else { 0.0 }).collect())
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
        for k in i..j {
            ranks[k] = avg_rank;
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

    // Compute autocorrelation at each lag using the FFT-free method
    let max_lag = n;
    let mut rho_hat = Vec::with_capacity(max_lag);

    for lag in 0..max_lag {
        let mut gamma = 0.0f64;
        for (ci, chain) in split.iter().enumerate() {
            let cm = chain_means[ci];
            let valid = n - lag;
            for t in 0..valid {
                gamma += (chain[t] - cm) * (chain[t + lag] - cm);
            }
        }
        gamma /= m_f * (n_f - 1.0);
        rho_hat.push(1.0 - (w - gamma) / w);
    }

    // Geyer's initial positive sequence: sum consecutive pairs until negative
    let mut tau = -1.0f64;
    let mut t = 1;
    while t + 1 < rho_hat.len() {
        let pair_sum = rho_hat[t] + rho_hat[t + 1];
        if pair_sum < 0.0 {
            break;
        }
        tau += pair_sum;
        t += 2;
    }
    tau = tau.max(1.0 / (m_f * n_f));

    m_f * n_f / (1.0 + 2.0 * tau)
}

fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut split = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        let mid = chain.len() / 2;
        split.push(chain[..mid].to_vec());
        split.push(chain[mid..].to_vec());
    }
    split
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

/// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm).
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -val
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(rh > 1.5, "R-hat should be large for diverged chains, got {}", rh);
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
}
