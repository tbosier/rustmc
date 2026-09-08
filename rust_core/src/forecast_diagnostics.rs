//! Parameter diagnostics for non-Hamiltonian forecasting samplers.
use crate::bayesian_ar::BayesianArPosterior;
use crate::bayesian_forecast::LocalLevelPosterior;
use crate::bayesian_seasonal::SeasonalLocalLevelPosterior;
use crate::bayesian_trend::LocalLinearTrendPosterior;
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

impl LocalLevelPosterior {
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let names =
            ["process_variance", "observation_variance", "terminal_level"].map(String::from);
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| vec![d.process_variance, d.observation_variance, d.terminal_level])
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }
}
impl LocalLinearTrendPosterior {
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let names = [
            "level_variance",
            "slope_variance",
            "observation_variance",
            "terminal_level",
            "terminal_slope",
        ]
        .map(String::from);
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        vec![
                            d.level_variance,
                            d.slope_variance,
                            d.observation_variance,
                            d.terminal_level,
                            d.terminal_slope,
                        ]
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }
}
impl SeasonalLocalLevelPosterior {
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let mut names = vec![
            "level_variance".into(),
            "seasonal_variance".into(),
            "observation_variance".into(),
            "terminal_level".into(),
        ];
        names.extend((1..self.period).map(|i| format!("terminal_seasonal[{}]", i - 1)));
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        let mut values = vec![
                            d.level_variance,
                            d.seasonal_variance,
                            d.observation_variance,
                        ];
                        values.extend_from_slice(&d.terminal_state);
                        values
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }
}
impl BayesianArPosterior {
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let mut names = vec!["intercept".into()];
        names.extend((1..=self.order).map(|i| format!("lag_{i}")));
        names.push("innovation_variance".into());
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|d| {
                        let mut values = d.coefficients.clone();
                        values.push(d.innovation_variance);
                        values
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }
}

impl crate::bayesian_regression::RegressionPosterior {
    /// Variances, regression coefficients and every terminal structural state.
    pub fn diagnostics(&self) -> DiagnosticsReport {
        let mut names = self.config.variance_names.clone();
        names.push("observation_variance".into());
        names.extend(
            (0..self.config.coefficient_prior.mean.len()).map(|i| format!("coefficient[{i}]")),
        );
        names.extend(
            (0..self.config.structural_model.dimension()).map(|i| format!("terminal_state[{i}]")),
        );
        let samples = self
            .chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|draw| {
                        let mut values = draw.variances.clone();
                        values.push(draw.observation_variance);
                        values.extend_from_slice(&draw.coefficients);
                        values.extend_from_slice(&draw.terminal_state);
                        values
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        parameter_diagnostics(&samples, &names)
    }
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
