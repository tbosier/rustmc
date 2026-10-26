//! Time one log-density gradient of a non-centered varying-intercept model.
//!
//! The model is compiled from the specification the Python builder produces
//! for a mean of `mu + tau * z["group"]`: `mu ~ Normal(0, 5)`,
//! `tau ~ HalfNormal(1)`, `z ~ Normal(0, 1)` per group, `sigma ~ HalfNormal(1)`,
//! and `y ~ Normal(mu + tau * z[group], sigma)`. Row `i` belongs to group
//! `(i * 7919) % groups`, so with more groups than rows some groups are
//! unused. For each row and group count it prints microseconds per gradient,
//! the median of several timed batches.
//!
//! A shape with a third field, `rows:groups:identity`, `rows:groups:exp` or
//! `rows:groups:mixed`, instead times a gather in which every row selects
//! group 0, with an identity or an `Exp` transform on every group, or an
//! identity on group 0 and `Exp` on the rest. It is the worst case for
//! transforming each group once: the rows read one group of many.
//!
//! ```text
//! cargo run --release -p rustmc_core --example indexed_gradient_cost [rows:groups[:transforms] ...]
//! ```

use rustmc_core::autodiff::Evaluator;
use rustmc_core::graph::{ElementwiseOp, Graph, ParamTransform};
use rustmc_core::model::{compile, HyperParam, LikelihoodFamily, LikelihoodSpec, ModelSpec};
use rustmc_core::model::{MuExpr, PriorSpec, SigmaSpec};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

fn spec(groups: usize) -> ModelSpec {
    let mu_expr = MuExpr::Add(
        Box::new(MuExpr::Param("mu".into())),
        Box::new(MuExpr::Binary(
            ElementwiseOp::Mul,
            Box::new(MuExpr::Param("tau".into())),
            Box::new(MuExpr::Gather {
                param_name: "z".into(),
                data_key: "group".into(),
            }),
        )),
    );
    ModelSpec {
        dimensions: HashMap::new(),
        potentials: Vec::new(),
        deterministics: Vec::new(),
        priors: vec![
            PriorSpec::Normal {
                name: "mu".into(),
                mu: HyperParam::Const(0.0),
                sigma: HyperParam::Const(5.0),
            },
            PriorSpec::HalfNormal {
                name: "tau".into(),
                sigma: HyperParam::Const(1.0),
            },
            PriorSpec::VectorNormal {
                name: "z".into(),
                n: groups,
                mu: 0.0,
                sigma: 1.0,
            },
            PriorSpec::HalfNormal {
                name: "sigma".into(),
                sigma: HyperParam::Const(1.0),
            },
        ],
        likelihoods: vec![LikelihoodSpec {
            family: LikelihoodFamily::Normal,
            name: "obs".into(),
            mu_expr,
            sigma: Some(SigmaSpec::Param("sigma".into())),
            observed_key: "y".into(),
        }],
        bound_data_1d: HashMap::new(),
        bound_data_2d: HashMap::new(),
    }
}

/// `mu + tau * z[group]` observed through a unit Normal, with every row in
/// group 0 and the transforms `transforms` names on the groups.
fn concentrated(rows: usize, groups: usize, transforms: &str) -> Graph {
    let mut g = Graph::new();
    let mu = g.add_param("mu");
    let tau = g.add_param("tau");
    let transform = match transforms {
        "identity" => ParamTransform::Identity,
        "exp" | "mixed" => ParamTransform::Exp,
        other => panic!("unknown transforms {other}"),
    };
    let start = g.add_vector_params_with_transform("z", groups, transform);
    if transforms == "mixed" {
        g.param_transforms[start] = ParamTransform::Identity;
    }
    let group = g.add_data("group", vec![0.0; rows]);
    let gathered = g.gather(start, groups, group);
    let scaled = g.elementwise(ElementwiseOp::Mul, tau, Some(gathered));
    let mean = g.scalar_broadcast_add(mu, scaled);
    let one = g.add_constant(1.0);
    let y = (0..rows)
        .map(|i| ((i * 104_729) % 1000) as f64 / 500.0 - 1.0)
        .collect();
    let obs = g.add_obs_data(y);
    g.obs_logp_normal(mean, one, obs);
    g
}

fn microseconds_per_gradient(rows: usize, groups: usize, transforms: Option<&str>) -> f64 {
    let graph = match transforms {
        Some(transforms) => concentrated(rows, groups, transforms),
        None => {
            let group = (0..rows).map(|i| ((i * 7919) % groups) as f64).collect();
            let y = (0..rows)
                .map(|i| ((i * 104_729) % 1000) as f64 / 500.0 - 1.0)
                .collect();
            let data = HashMap::from([("group".to_string(), group), ("y".to_string(), y)]);
            compile(&spec(groups), &data, &HashMap::new())
                .expect("the model compiles")
                .graph
        }
    };
    let mut evaluator = Evaluator::new(&graph);
    let mut position = (0..graph.param_count)
        .map(|k| 0.05 * (k % 11) as f64 - 0.2)
        .collect::<Vec<_>>();
    let calls = 2_000;
    let mut batches = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for call in 0..calls {
            position[0] = 0.01 * (call % 7) as f64;
            evaluator.compute(&graph, black_box(&position));
            black_box(&evaluator.grad);
        }
        batches.push(start.elapsed().as_secs_f64() * 1e6 / calls as f64);
    }
    batches.sort_by(f64::total_cmp);
    batches[batches.len() / 2]
}

fn main() {
    let mut shapes = std::env::args()
        .skip(1)
        .map(|shape| {
            let mut fields = shape.split(':');
            let mut number = |what: &str| {
                fields
                    .next()
                    .and_then(|field| field.parse().ok())
                    .unwrap_or_else(|| panic!("{what} is a positive integer in rows:groups"))
            };
            let (rows, groups) = (number("rows"), number("groups"));
            (rows, groups, fields.next().map(str::to_string))
        })
        .collect::<Vec<(usize, usize, Option<String>)>>();
    if shapes.is_empty() {
        shapes = vec![(500, 20, None), (5000, 100, None)];
    }
    println!("varying intercept; microseconds per gradient (median of 9 batches)");
    for (rows, groups, transforms) in shapes {
        println!(
            "{:>6} rows {:>5} groups {:>9} {:>9.2}",
            rows,
            groups,
            transforms.as_deref().unwrap_or(""),
            microseconds_per_gradient(rows, groups, transforms.as_deref())
        );
    }
}
