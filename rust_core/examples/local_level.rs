use rustmc_core::state_space::LinearGaussianStateSpace;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LinearGaussianStateSpace::local_level(
        0.25, // process variance
        1.0,  // observation variance
        0.0,  // initial level mean
        10.0, // initial level variance
    )?;

    let filtered = model.filter(&[10.0, 10.5, 11.0, 10.8])?;
    println!("log likelihood: {}", filtered.log_likelihood);
    Ok(())
}
