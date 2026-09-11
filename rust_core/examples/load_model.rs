//! Load a Python-authored artifact without Python. See docs/native-models.md.
use rustmc_core::data::{DataInputs, MatrixBinding};
use rustmc_core::model::GraphModel;
use rustmc_core::sampler::SamplerConfig;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

fn inputs(text: &str) -> Result<DataInputs, Box<dyn Error>> {
    let object: HashMap<String, Value> = serde_json::from_str(text)?;
    let mut input = DataInputs::default();
    for (key, value) in object {
        let rows = value.as_array().ok_or("data values must be arrays")?;
        if rows.first().is_some_and(Value::is_array) {
            let matrix: Vec<Vec<f64>> = serde_json::from_value(value)?;
            let n_cols = matrix[0].len();
            if matrix.iter().any(|r| r.len() != n_cols) {
                return Err("matrix rows must have equal lengths".into());
            }
            input.matrices.insert(
                key,
                MatrixBinding {
                    n_rows: matrix.len(),
                    n_cols,
                    data: Arc::from(matrix.into_iter().flatten().collect::<Vec<_>>()),
                },
            );
        } else {
            let vector: Vec<f64> = serde_json::from_value(value)?;
            input.vectors.insert(key, Arc::from(vector));
        }
    }
    Ok(input)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: load_model MODEL.json DATA.json POSITION.json FUTURE.json".into());
    }
    let model = GraphModel::from_json(&std::fs::read_to_string(&args[0])?)?;
    let data = inputs(&std::fs::read_to_string(&args[1])?)?;
    let binding = model.bind(data, "dataset")?;
    let position: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let (density, gradient) = model.log_density(&binding, &position)?;
    let fit = model.sample(
        binding,
        SamplerConfig {
            num_chains: 2,
            num_warmup: 500,
            num_draws: 500,
            target_accept: 0.95,
            seed: 42,
            show_progress: false,
            ..Default::default()
        },
        None,
    )?;
    let future = inputs(&std::fs::read_to_string(&args[3])?)?;
    let predictions = fit.predict(future, HashMap::new(), 43, true)?;
    println!(
        "{}",
        json!({"log_density": density, "gradient": gradient,
        "param_names": fit.samples.param_names, "posterior_mean": fit.samples.mean(),
        "predictions": predictions, "artifact": serde_json::from_str::<Value>(&model.to_json()?)?})
    );
    Ok(())
}
