//! Time one gradient of graphs with many Normal terms, to show how the cost
//! grows with the number of terms: grouped terms `mu + tau * z_k[group_k]`,
//! which the evaluator can fuse, and plain terms `mu + x_k` on a data column,
//! which it cannot. Each term has its own 20 rows; grouped terms have their
//! own 5 group effects. All terms share `mu`, `tau` and `sigma`. For each kind
//! and term count it prints microseconds per gradient and per term, and the
//! fused reverse passes per gradient.
//!
//! ```text
//! cargo run --release -p rustmc_core --example grouped_scaling [terms...]
//! ```

use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::{ElementwiseOp, Graph};
use std::hint::black_box;
use std::time::Instant;

const ROWS: usize = 20;
const GROUPS: usize = 5;

fn graph(terms: usize, grouped: bool) -> Graph {
    let mut g = Graph::new();
    let mu = g.add_param("mu");
    let zero = g.add_constant(0.0);
    let five = g.add_constant(5.0);
    let one = g.add_constant(1.0);
    g.normal_logp(mu, zero, five);
    let raw_tau = g.add_param("log_tau");
    let tau = g.exp(raw_tau);
    g.log_half_normal_logp(raw_tau, one);
    let raw_sigma = g.add_param("log_sigma");
    let sigma = g.exp(raw_sigma);
    g.log_half_normal_logp(raw_sigma, one);
    for k in 0..terms {
        let y = (0..ROWS)
            .map(|i| ((i * 7 + k * 3) % 11) as f64 / 5.0 - 1.0)
            .collect();
        let mean = if grouped {
            let z = g.add_vector_params(&format!("z{k}"), GROUPS);
            g.vector_normal_logp(z, GROUPS, 0.0, 1.0);
            let group = g.add_data(
                &format!("group{k}"),
                (0..ROWS).map(|i| ((i + k) % GROUPS) as f64).collect(),
            );
            let gathered = g.gather(z, GROUPS, group);
            let product = g.elementwise(ElementwiseOp::Mul, tau, Some(gathered));
            g.scalar_broadcast_add(mu, product)
        } else {
            let x = g.add_data(
                &format!("x{k}"),
                (0..ROWS)
                    .map(|i| ((i * 5 + k) % 13) as f64 / 6.0 - 1.0)
                    .collect(),
            );
            g.scalar_broadcast_add(mu, x)
        };
        let obs = g.add_obs_data(y);
        g.obs_logp_normal(mean, sigma, obs);
    }
    g
}

fn main() {
    let mut counts = std::env::args()
        .skip(1)
        .map(|count| {
            count
                .parse()
                .expect("each term count is a positive integer")
        })
        .collect::<Vec<usize>>();
    if counts.is_empty() {
        counts = vec![10, 100, 1000];
    }
    println!("microseconds per gradient (median of 9 batches) and per term");
    for grouped in [true, false] {
        for &terms in &counts {
            let g = graph(terms, grouped);
            let mut evaluator = Evaluator::new(&g);
            let mut position = (0..g.param_count)
                .map(|k| 0.05 * (k % 11) as f64 - 0.2)
                .collect::<Vec<_>>();
            let calls = (200_000 / terms).max(20);
            let mut batches = Vec::new();
            for _ in 0..9 {
                let start = Instant::now();
                for call in 0..calls {
                    position[0] = 0.01 * (call % 7) as f64;
                    evaluator.compute(&g, black_box(&position));
                    black_box(&evaluator.grad);
                }
                batches.push(start.elapsed().as_secs_f64() * 1e6 / calls as f64);
            }
            let before = fused_reverse_passes();
            drop(evaluator);
            let fused = (fused_reverse_passes() - before) as f64 / (9 * calls) as f64;
            batches.sort_by(f64::total_cmp);
            let microseconds = batches[4];
            println!(
                "{:>8} {:>5} terms {:>10.2} {:>8.3} per term   fused reverse passes per gradient {:.0}",
                if grouped { "grouped" } else { "plain" },
                terms,
                microseconds,
                microseconds / terms as f64,
                fused
            );
        }
    }
}

/// The process-wide count of fused grouped-Normal reverse passes.
fn fused_reverse_passes() -> u64 {
    rustmc_core::autodiff::fused_grouped_normal_passes().1
}
