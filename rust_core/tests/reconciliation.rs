//! Forecast reconciliation against the textbook formula and a known process.
//!
//! The reference here forms `P = (S' W^-1 S)^-1 S' W^-1` with dense inverses,
//! which the library deliberately never does, so the two share no algebra.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rustmc_core::reconciliation::{reconcile, Hierarchy, NodeDraws, ReconciliationMethod};

type Matrix = Vec<Vec<f64>>;

fn transpose(a: &Matrix) -> Matrix {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .map(|row| {
            (0..b[0].len())
                .map(|j| row.iter().zip(b).map(|(x, b_row)| x * b_row[j]).sum())
                .collect()
        })
        .collect()
}

/// Gauss-Jordan inverse with partial pivoting.
fn inverse(a: &Matrix) -> Matrix {
    let n = a.len();
    let mut work: Matrix = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut extended = row.clone();
            extended.extend((0..n).map(|j| f64::from(u8::from(i == j))));
            extended
        })
        .collect();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&x, &y| work[x][column].abs().total_cmp(&work[y][column].abs()))
            .unwrap();
        work.swap(column, pivot);
        let scale = work[column][column];
        assert!(scale.abs() > 1e-300, "singular reference matrix");
        for value in &mut work[column] {
            *value /= scale;
        }
        for row in 0..n {
            if row != column {
                let factor = work[row][column];
                let pivot_row = work[column].clone();
                for (value, pivot_value) in work[row].iter_mut().zip(pivot_row) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }
    work.into_iter().map(|row| row[n..].to_vec()).collect()
}

/// `S (S' W^-1 S)^-1 S' W^-1`, the reconciliation map on all nodes.
fn textbook_map(s: &Matrix, w: &Matrix) -> Matrix {
    let w_inverse = inverse(w);
    let st = transpose(s);
    let st_w = multiply(&st, &w_inverse);
    let p = multiply(&inverse(&multiply(&st_w, s)), &st_w);
    multiply(s, &p)
}

fn diagonal(values: &[f64]) -> Matrix {
    (0..values.len())
        .map(|i| {
            (0..values.len())
                .map(|j| if i == j { values[i] } else { 0.0 })
                .collect()
        })
        .collect()
}

/// The MinT shrinkage covariance, transcribed from `hts::shrink.estim`.
fn shrunk_covariance(residuals: &Matrix) -> (Matrix, f64) {
    let t = residuals.len() as f64;
    let n = residuals[0].len();
    let covariance: Matrix = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| residuals.iter().map(|row| row[i] * row[j]).sum::<f64>() / t)
                .collect()
        })
        .collect();
    let sd: Vec<f64> = (0..n).map(|i| covariance[i][i].sqrt()).collect();
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let w: Vec<f64> = residuals
                    .iter()
                    .map(|row| row[i] / sd[i] * row[j] / sd[j])
                    .collect();
                let mean = w.iter().sum::<f64>() / t;
                numerator += w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((t - 1.0) * t);
                denominator += mean * mean;
            }
        }
    }
    let lambda = (numerator / denominator).clamp(0.0, 1.0);
    let shrunk = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        covariance[i][j]
                    } else {
                        (1.0 - lambda) * covariance[i][j]
                    }
                })
                .collect()
        })
        .collect();
    (shrunk, lambda)
}

/// total -> 3 regions -> 4 stores each: 12 bottom series, 16 nodes.
fn stores() -> Hierarchy {
    let labels: Vec<Vec<String>> = (0..12)
        .map(|store| vec![format!("R{}", store / 4), format!("s{store}")])
        .collect();
    Hierarchy::from_labels(&labels, "total", "/").unwrap()
}

fn grid() -> Hierarchy {
    // Total, two products, two regions over a 2 x 2 grid: a grouped structure.
    let rows = vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    Hierarchy::from_summing_matrix(&rows, None).unwrap()
}

fn normals(rng: &mut ChaCha8Rng, count: usize) -> Vec<f64> {
    (0..count).map(|_| StandardNormal.sample(rng)).collect()
}

/// Correlated, strongly heteroskedastic residuals for `n` nodes.
fn residual_rows(rng: &mut ChaCha8Rng, times: usize, n: usize) -> Matrix {
    (0..times)
        .map(|_| {
            let common: f64 = StandardNormal.sample(rng);
            (0..n)
                .map(|i| {
                    let own: f64 = StandardNormal.sample(rng);
                    (0.5 + i as f64 * 0.4) * (0.6 * common + own)
                })
                .collect()
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let tolerance = 1e-10 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn projections_match_the_textbook_formula() {
    for hierarchy in [stores(), grid()] {
        let n = hierarchy.node_count();
        let s = hierarchy.summing_matrix().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let residuals = residual_rows(&mut rng, 30, n);
        let (chains, draws, steps) = (2, 5, 3);
        let base = NodeDraws::new(
            normals(&mut rng, chains * draws * n * steps)
                .into_iter()
                .map(|z| 10.0 + 4.0 * z)
                .collect(),
            chains,
            draws,
            n,
            steps,
        )
        .unwrap();
        let structural: Vec<f64> = s.iter().map(|row| row.iter().sum()).collect();
        let mean_squares: Vec<f64> = (0..n)
            .map(|i| residuals.iter().map(|row| row[i] * row[i]).sum::<f64>() / 30.0)
            .collect();
        let (shrunk, lambda) = shrunk_covariance(&residuals);
        assert!(lambda > 0.0 && lambda < 1.0, "lambda {lambda}");
        let bottom_up = {
            let mut p = vec![vec![0.0; n]; hierarchy.bottom_count()];
            for (j, row) in p.iter_mut().enumerate() {
                row[hierarchy.aggregate_count() + j] = 1.0;
            }
            multiply(&s, &p)
        };
        let cases = [
            (ReconciliationMethod::BottomUp, bottom_up),
            (
                ReconciliationMethod::Ols,
                textbook_map(&s, &diagonal(&vec![1.0; n])),
            ),
            (
                ReconciliationMethod::WlsStructural,
                textbook_map(&s, &diagonal(&structural)),
            ),
            (
                ReconciliationMethod::WlsVariance {
                    residuals: residuals.clone(),
                },
                textbook_map(&s, &diagonal(&mean_squares)),
            ),
            (
                ReconciliationMethod::MinTShrink {
                    residuals: residuals.clone(),
                },
                textbook_map(&s, &shrunk),
            ),
        ];
        for (method, map) in cases {
            let result = reconcile(&hierarchy, &base, &method).unwrap();
            if let Some(intensity) = result.shrinkage_intensity {
                assert_close(intensity, lambda, "shrinkage intensity");
            }
            let values = result.draws.values();
            for sample in 0..chains * draws {
                for step in 0..steps {
                    let at = |node: usize| sample * n * steps + node * steps + step;
                    for (node, row) in map.iter().enumerate() {
                        let expected: f64 = row
                            .iter()
                            .enumerate()
                            .map(|(k, weight)| weight * base.values()[at(k)])
                            .sum();
                        assert_close(values[at(node)], expected, method.name());
                    }
                }
            }
        }
    }
}

#[test]
fn every_reconciled_draw_is_coherent_and_keeps_its_identity() {
    let hierarchy = stores();
    let n = hierarchy.node_count();
    let (m, r) = (hierarchy.bottom_count(), hierarchy.aggregate_count());
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let residuals = residual_rows(&mut rng, 40, n);
    let (chains, draws, steps) = (3, 40, 4);
    // Positive, clearly incoherent draws with a scale that grows with the node.
    let values: Vec<f64> = normals(&mut rng, chains * draws * n * steps)
        .into_iter()
        .enumerate()
        .map(|(index, z)| 50.0 * (1.0 + (index / steps % n) as f64) + 5.0 * z.abs())
        .collect();
    let base = NodeDraws::new(values, chains, draws, n, steps).unwrap();
    let history: Vec<Vec<f64>> = (0..20)
        .map(|t| {
            (0..m)
                .map(|j| 1.0 + ((t * 7 + j * 3) % 11) as f64)
                .collect()
        })
        .collect();
    let methods = [
        ReconciliationMethod::BottomUp,
        ReconciliationMethod::TopDownAverageProportions { history },
        ReconciliationMethod::TopDownForecastProportions,
        ReconciliationMethod::Ols,
        ReconciliationMethod::WlsStructural,
        ReconciliationMethod::WlsVariance {
            residuals: residuals.clone(),
        },
        ReconciliationMethod::MinTShrink { residuals },
    ];
    // Reverse the draw order of every chain; projections act per draw, so the
    // reconciled draws must come back reversed too.
    let reversed = {
        let stride = n * steps;
        let mut values = Vec::with_capacity(base.values().len());
        for chain in 0..chains {
            for draw in (0..draws).rev() {
                let start = (chain * draws + draw) * stride;
                values.extend_from_slice(&base.values()[start..start + stride]);
            }
        }
        NodeDraws::new(values, chains, draws, n, steps).unwrap()
    };
    for method in methods {
        let result = reconcile(&hierarchy, &base, &method).unwrap();
        assert_eq!(result.draws.shape(), base.shape());
        let values = result.draws.values();
        let mut worst: f64 = 0.0;
        for sample in 0..chains * draws {
            for step in 0..steps {
                let at = |node: usize| values[sample * n * steps + node * steps + step];
                for i in 0..r {
                    let sum: f64 = hierarchy.support(i).iter().map(|&j| at(r + j)).sum();
                    worst = worst.max((at(i) - sum).abs() / sum.abs().max(1.0));
                }
            }
        }
        assert!(
            worst <= 1e-10,
            "{}: relative incoherence {worst}",
            method.name()
        );
        // Forecast proportions come from the base means, whose last bits
        // depend on the order the draws are summed in; every other method maps
        // each draw on its own and must match exactly.
        let exact = !matches!(method, ReconciliationMethod::TopDownForecastProportions);
        let permuted = reconcile(&hierarchy, &reversed, &method).unwrap();
        let stride = n * steps;
        for chain in 0..chains {
            for draw in 0..draws {
                let original = &values[(chain * draws + draw) * stride..][..stride];
                let moved = (chain * draws + draws - 1 - draw) * stride;
                let moved = &permuted.draws.values()[moved..moved + stride];
                if exact {
                    assert_eq!(original, moved, "{}", method.name());
                } else {
                    for (a, b) in original.iter().zip(moved) {
                        assert!((a - b).abs() <= 1e-12 * a.abs(), "{}", method.name());
                    }
                }
            }
        }
    }
}

/// Continuous ranked probability score of an ensemble, by the sorted-sample
/// form `2/n^2 sum_i (x_(i) - y) (n 1{y < x_(i)} - i + 1/2)`.
fn crps(mut draws: Vec<f64>, actual: f64) -> f64 {
    draws.sort_by(f64::total_cmp);
    let n = draws.len() as f64;
    2.0 / (n * n)
        * draws
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let above = if actual < x { n } else { 0.0 };
                (x - actual) * (above - (i as f64 + 1.0) + 0.5)
            })
            .sum::<f64>()
}

/// Base forecast errors for the stores hierarchy: one shared factor, one
/// factor per region, and node noise whose scale varies thirtyfold. Aggregate
/// errors are not the sums of bottom errors, so base forecasts are incoherent.
struct ErrorProcess {
    loadings: Matrix,
    scales: Vec<f64>,
}

impl ErrorProcess {
    fn new(hierarchy: &Hierarchy) -> Self {
        let n = hierarchy.node_count();
        let r = hierarchy.aggregate_count();
        let scales: Vec<f64> = (0..n)
            .map(|node| match node {
                0 => 6.0,
                1..=3 => 3.0,
                _ => [0.2, 0.5, 1.5, 6.0][(node - r) % 4],
            })
            .collect();
        let loadings = (0..n)
            .map(|node| {
                let region = if node == 0 {
                    None
                } else if node < r {
                    Some(node - 1)
                } else {
                    Some((node - r) / 4)
                };
                (0..4)
                    .map(|factor| match factor {
                        0 => 0.6,
                        k if Some(k - 1) == region => 0.6,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        Self { loadings, scales }
    }

    fn sample(&self, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let factors = normals(rng, 4);
        self.loadings
            .iter()
            .zip(&self.scales)
            .map(|(loading, scale)| {
                let own: f64 = StandardNormal.sample(rng);
                let shared: f64 = loading.iter().zip(&factors).map(|(l, f)| l * f).sum();
                scale * (shared + own)
            })
            .collect()
    }

    fn covariance(&self) -> Matrix {
        let n = self.scales.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let shared: f64 = self.loadings[i]
                            .iter()
                            .zip(&self.loadings[j])
                            .map(|(a, b)| a * b)
                            .sum();
                        let own = if i == j { 1.0 } else { 0.0 };
                        self.scales[i] * self.scales[j] * (shared + own)
                    })
                    .collect()
            })
            .collect()
    }
}

#[test]
fn mint_and_wls_beat_incoherent_base_forecasts_on_a_known_process() {
    let hierarchy = stores();
    let n = hierarchy.node_count();
    let s = hierarchy.summing_matrix().unwrap();
    let process = ErrorProcess::new(&hierarchy);
    let covariance = process.covariance();
    let mut rng = ChaCha8Rng::seed_from_u64(20260927);
    let (replications, ensemble, times) = (1000, 200, 60);

    // Oracle MinT, with the true covariance: the lowest achievable squared
    // error of any linear unbiased reconciliation, per node.
    let oracle = textbook_map(&s, &covariance);
    let oracle_mse = multiply(&multiply(&oracle, &covariance), &transpose(&oracle))
        .iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .sum::<f64>()
        / n as f64;
    let base_mse_theory = (0..n).map(|i| covariance[i][i]).sum::<f64>() / n as f64;

    let names = ["base", "ols", "wls_struct", "wls_var", "mint_shrink"];
    let mut squared = [0.0; 5];
    let mut scores = [0.0; 5];
    for _ in 0..replications {
        // A coherent truth; the base mean misses it by one error draw.
        let bottom: Vec<f64> = (0..hierarchy.bottom_count())
            .map(|j| 20.0 + j as f64)
            .collect();
        let truth: Vec<f64> = s
            .iter()
            .map(|row| row.iter().zip(&bottom).map(|(a, b)| a * b).sum())
            .collect();
        let miss = process.sample(&mut rng);
        let center: Vec<f64> = truth.iter().zip(&miss).map(|(t, e)| t + e).collect();
        // Joint base draws: every draw is one sample of all nodes together.
        let values: Vec<f64> = (0..ensemble)
            .flat_map(|_| {
                let error = process.sample(&mut rng);
                center
                    .iter()
                    .zip(error)
                    .map(|(c, e)| c + e)
                    .collect::<Vec<_>>()
            })
            .collect();
        let base = NodeDraws::new(values, 1, ensemble, n, 1).unwrap();
        let residuals: Matrix = (0..times).map(|_| process.sample(&mut rng)).collect();
        let methods = [
            None,
            Some(ReconciliationMethod::Ols),
            Some(ReconciliationMethod::WlsStructural),
            Some(ReconciliationMethod::WlsVariance {
                residuals: residuals.clone(),
            }),
            Some(ReconciliationMethod::MinTShrink { residuals }),
        ];
        for (index, method) in methods.iter().enumerate() {
            let draws = match method {
                None => base.clone(),
                Some(method) => reconcile(&hierarchy, &base, method).unwrap().draws,
            };
            let means = draws.means();
            for node in 0..n {
                squared[index] += (means[node] - truth[node]).powi(2);
                let column: Vec<f64> = (0..ensemble)
                    .map(|draw| draws.values()[draw * n + node])
                    .collect();
                scores[index] += crps(column, truth[node]);
            }
        }
    }
    let per_node = (replications * n) as f64;
    let mse: Vec<f64> = squared.iter().map(|value| value / per_node).collect();
    let crps: Vec<f64> = scores.iter().map(|value| value / per_node).collect();
    for (index, name) in names.iter().enumerate() {
        eprintln!(
            "{name:>12}: MSE {:.3} ({:.3} of base), CRPS {:.3} ({:.3} of base)",
            mse[index],
            mse[index] / mse[0],
            crps[index],
            crps[index] / crps[0]
        );
    }
    eprintln!("theory: base MSE {base_mse_theory:.3}, oracle MinT MSE {oracle_mse:.3}");

    // The base forecasts are what they claim to be: their error variance is
    // the process's, within sampling error (1000 x 16 errors).
    assert!((mse[0] / base_mse_theory - 1.0).abs() < 0.06, "{mse:?}");
    // MinT with estimated weights comes within 20% of the oracle optimum.
    // Over seeds 1000..=6000 it landed 4-11% above it, WLS with variance
    // scaling (right variances, no correlations) 26-34% above, and OLS about
    // 2.2 times, so a projection that drops the correlations or ignores the
    // variances fails here. This does not check the shrinkage intensity: in
    // review, the Python version of this test passed with the intensity fixed
    // at 0, 0.25, 0.5 or 0.75. projections_match_the_textbook_formula checks it.
    assert!(mse[4] < 1.2 * oracle_mse, "{mse:?} vs {oracle_mse}");
    assert!(mse[3] > 1.2 * oracle_mse, "{mse:?} vs {oracle_mse}");
    assert!(mse[1] > 1.5 * oracle_mse, "{mse:?} vs {oracle_mse}");
    // Against the incoherent base forecasts, with margins far wider than the
    // spread across those seeds.
    assert!(mse[4] < 0.45 * mse[0], "{mse:?}");
    assert!(mse[3] < 0.55 * mse[0], "{mse:?}");
    assert!(crps[4] < 0.72 * crps[0], "{crps:?}");
    assert!(crps[3] < 0.76 * crps[0], "{crps:?}");
    assert!(crps[4] < 0.9 * crps[1], "{crps:?}");
    // Structural scaling assumes every bottom series has the same error
    // variance. Here they differ thirtyfold, and it is worse than doing
    // nothing: a negative result the documentation states.
    assert!(mse[2] > mse[0] && crps[2] > crps[0], "{mse:?} {crps:?}");
}
