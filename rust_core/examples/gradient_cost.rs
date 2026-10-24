//! Time one log-density gradient of a regression on a design matrix.
//!
//! A NUTS fit of a small model spends nearly all its time in these
//! gradients, so this is the number to watch when changing how the
//! evaluator computes them. For each likelihood and design width it prints
//! microseconds per gradient, the median of several timed batches.
//!
//! ```text
//! cargo run --release -p rustmc_core --example gradient_cost [rows [widths...]]
//! ```

use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::Graph;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum Likelihood {
    Normal,
    NormalWithInferredScale,
    BernoulliLogit,
    PoissonLog,
}

fn graph(likelihood: Likelihood, rows: usize, columns: usize) -> (Graph, usize) {
    let mut g = Graph::new();
    let start = g.add_vector_params("beta", columns);
    // `vector_normal_logp` adds itself to the log density.
    g.vector_normal_logp(start, columns, 0.0, 1.0);
    let design = (0..rows * columns)
        .map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5)
        .collect();
    let matrix = g.store_matrix(design, rows, columns);
    let eta = g.mat_vec_mul(matrix, start, columns, None);
    let mut parameters = columns;
    let response = |i: usize| ((i * 104_729) % 1000) as f64 / 1000.0;
    match likelihood {
        Likelihood::Normal | Likelihood::NormalWithInferredScale => {
            let sigma = if matches!(likelihood, Likelihood::Normal) {
                g.add_constant(1.0)
            } else {
                parameters += 1;
                let raw = g.add_param("log_sigma");
                g.exp(raw)
            };
            let obs = g.add_obs_data((0..rows).map(|i| 2.0 * response(i) - 1.0).collect());
            g.obs_logp_normal(eta, sigma, obs);
        }
        Likelihood::BernoulliLogit => {
            let obs = g.add_obs_data(
                (0..rows)
                    .map(|i| (response(i) > 0.5) as u8 as f64)
                    .collect(),
            );
            g.obs_logp_bernoulli_logit(eta, obs);
        }
        Likelihood::PoissonLog => {
            let obs = g.add_obs_data((0..rows).map(|i| (4.0 * response(i)).floor()).collect());
            g.obs_logp_poisson_log(eta, obs);
        }
    }
    (g, parameters)
}

fn microseconds_per_gradient(g: &Graph, parameters: usize) -> f64 {
    let mut evaluator = Evaluator::new(g);
    let mut position = (0..parameters)
        .map(|k| 0.05 * k as f64 - 0.1)
        .collect::<Vec<_>>();
    // Enough calls per batch for about 20 ms of work at the slowest width.
    let calls = 2_000;
    let mut batches = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for call in 0..calls {
            position[0] = 0.01 * (call % 7) as f64;
            evaluator.compute(g, black_box(&position));
            black_box(&evaluator.grad);
        }
        batches.push(start.elapsed().as_secs_f64() * 1e6 / calls as f64);
    }
    batches.sort_by(f64::total_cmp);
    batches[batches.len() / 2]
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let rows = arguments
        .next()
        .map(|value| value.parse().expect("rows is a positive integer"))
        .unwrap_or(500);
    let mut widths = arguments
        .map(|value| value.parse().expect("each width is a positive integer"))
        .collect::<Vec<usize>>();
    if widths.is_empty() {
        widths = vec![2, 8, 32, 128, 512];
    }
    println!("{rows} rows; microseconds per gradient (median of 9 batches)");
    let header = widths
        .iter()
        .map(|w| format!("{:>8}", format!("p={w}")))
        .collect::<String>();
    println!("{:<28}{header}", "likelihood");
    for likelihood in [
        Likelihood::Normal,
        Likelihood::NormalWithInferredScale,
        Likelihood::BernoulliLogit,
        Likelihood::PoissonLog,
    ] {
        let cells = widths
            .iter()
            .map(|&columns| {
                let (g, parameters) = graph(likelihood, rows, columns);
                format!("{:>8.2}", microseconds_per_gradient(&g, parameters))
            })
            .collect::<String>();
        println!("{:<28}{cells}", format!("{likelihood:?}"));
    }
}
