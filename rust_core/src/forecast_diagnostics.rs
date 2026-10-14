//! Parameter diagnostics for non-Hamiltonian forecasting samplers.
//!
//! Each posterior type builds its own parameter names and samples, next to
//! its model, and passes them to [`parameter_diagnostics`].
use crate::diagnostics::{compute_diagnostics, DiagnosticsReport};

/// The fewest retained draws per chain for which [`parameter_diagnostics`]
/// reports R-hat, bulk/tail ESS and the MCSE: two per split half, the rule
/// [`compute_diagnostics`] applies to every engine.
pub const MIN_DIAGNOSTIC_DRAWS_PER_CHAIN: usize = 4;

/// Reuse the rank-normalized folded split diagnostics without fabricating
/// Hamiltonian sampler telemetry.
///
/// Availability follows the one rule [`compute_diagnostics`] applies to every
/// engine, HMC/NUTS fits included: an empty or ragged sample, or a parameter
/// with a non-finite draw, is unavailable; R-hat, bulk/tail ESS and the MCSE
/// need at least four draws per chain (two per split half), and a parameter
/// whose draws never move has none of them. Unavailable values are NaN in
/// Rust and None in the specialised Python APIs.
pub fn parameter_diagnostics(samples: &[Vec<Vec<f64>>], names: &[String]) -> DiagnosticsReport {
    compute_diagnostics(samples, names, &[], 0)
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
    fn forecasting_fits_share_the_engine_wide_availability_rule() {
        // These forecasting fits used to blank R-hat, ESS and MCSE below six
        // draws per chain, while the same draws from an HMC fit reported them
        // from four. Both now follow compute_diagnostics: HMC/NUTS fits reach
        // it through SampleResult::diagnostics with their draws unchanged, so
        // the bit comparison below fails if the forecasting path adds a floor
        // of its own or reorders the draws; the availability assertions pin
        // the floor itself.
        let names = ["process_variance", "observation_variance", "terminal_level"]
            .map(String::from)
            .to_vec();
        for draws in [3usize, 4, 5, 6, 40] {
            let posterior = crate::bayesian_forecast::LocalLevelPosterior {
                chains: (0..2)
                    .map(|chain| {
                        (0..draws)
                            .map(|draw| {
                                let x = ((chain * 31 + draw * 7) % 11) as f64;
                                crate::bayesian_forecast::LocalLevelPosteriorDraw {
                                    process_variance: 0.1 + x,
                                    observation_variance: 2.0,
                                    terminal_level: x - 5.0,
                                }
                            })
                            .collect()
                    })
                    .collect(),
            };
            let samples: Vec<Vec<Vec<f64>>> = posterior
                .chains
                .iter()
                .map(|chain| {
                    chain
                        .iter()
                        .map(|d| vec![d.process_variance, d.observation_variance, d.terminal_level])
                        .collect()
                })
                .collect();
            let forecasting = posterior.diagnostics();
            let engine = compute_diagnostics(&samples, &names, &[], 0);
            for (ours, theirs) in forecasting.params.iter().zip(&engine.params) {
                for (a, b) in [
                    (ours.r_hat, theirs.r_hat),
                    (ours.ess_bulk, theirs.ess_bulk),
                    (ours.ess_tail, theirs.ess_tail),
                    (ours.mcse_mean, theirs.mcse_mean),
                ] {
                    assert_eq!(a.to_bits(), b.to_bits(), "{draws} draws: {ours:?}");
                }
            }
            // The moving parameters are available from four draws per chain;
            // the constant one never is.
            let moving = &forecasting.params[0];
            assert_eq!(
                moving.r_hat.is_finite(),
                draws >= MIN_DIAGNOSTIC_DRAWS_PER_CHAIN,
                "{draws}: {moving:?}"
            );
            assert_eq!(
                moving.ess_bulk.is_finite(),
                draws >= MIN_DIAGNOSTIC_DRAWS_PER_CHAIN
            );
            assert!(forecasting.params[1].r_hat.is_nan());
            assert!(forecasting.params[1].ess_bulk.is_nan());
        }
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
