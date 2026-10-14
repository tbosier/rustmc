//! Parameter diagnostics for non-Hamiltonian forecasting samplers.
//!
//! Each posterior type builds its own parameter names and samples, next to
//! its model, and passes them to [`parameter_diagnostics`].
use crate::diagnostics::{compute_diagnostics, DiagnosticsReport};

/// Reuse the rank-normalized folded split diagnostics without fabricating
/// Hamiltonian sampler telemetry. Nonfinite or insufficient/constant traces have
/// unavailable convergence metrics (NaN in Rust, None in specialized Python APIs).
pub fn parameter_diagnostics(samples: &[Vec<Vec<f64>>], names: &[String]) -> DiagnosticsReport {
    let mut report = compute_diagnostics(samples, names, &[], 0);
    // The shared engine rejects empty/ragged parameter matrices before ranking.
    if samples
        .iter()
        .flatten()
        .any(|draw| draw.len() != names.len())
    {
        return report;
    }
    for (index, parameter) in report.params.iter_mut().enumerate() {
        let first = samples
            .first()
            .and_then(|chain| chain.first())
            .map(|draw| draw[index]);
        let unavailable = samples.is_empty()
            || samples.iter().any(|chain| chain.len() < 6)
            || samples
                .iter()
                .flatten()
                .any(|draw| !draw[index].is_finite())
            || first.is_some_and(|first| samples.iter().flatten().all(|draw| draw[index] == first));
        if unavailable {
            parameter.r_hat = f64::NAN;
            parameter.ess_bulk = f64::NAN;
            parameter.ess_tail = f64::NAN;
            parameter.mcse_mean = f64::NAN;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_and_nonfinite_samples_are_unavailable_without_panics_or_hangs() {
        let invalid = vec![
            vec![],
            vec![vec![]],
            vec![vec![vec![1.]], vec![vec![1.]; 3]],
            vec![vec![vec![]; 8]],
            vec![vec![vec![f64::NAN]; 8]],
            vec![vec![vec![f64::INFINITY]; 8]],
        ];
        for samples in invalid {
            let report = parameter_diagnostics(&samples, &["x".into()]);
            assert_eq!(report.params.len(), 1);
            assert!(report.params[0].r_hat.is_nan());
            assert!(report.params[0].ess_bulk.is_nan());
        }
    }
    #[test]
    fn separated_constant_chains_retain_infinite_rhat() {
        let samples = vec![vec![vec![0.0]; 20], vec![vec![10.0]; 20]];
        let report = parameter_diagnostics(&samples, &["x".into()]);
        assert!(report.params[0].r_hat.is_infinite());
    }
    #[test]
    fn unavailable_short_and_constant_traces() {
        for samples in [vec![vec![vec![1.]; 3]; 2], vec![vec![vec![1.]; 40]; 2]] {
            let report = parameter_diagnostics(&samples, &["x".into()]);
            assert!(report.params[0].r_hat.is_nan());
            assert!(report.params[0].ess_bulk.is_nan());
            assert!(report.params[0].mcse_mean.is_nan());
            assert!(!report
                .to_table_with_sampler(Some("Sampler: Gibbs"))
                .contains("Mean accept rate"));
        }
    }
}
