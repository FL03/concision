/*
    Appellation: linear <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::{linarr, Predict, Result};
use cnc::func::Sigmoid;
use cnc::linear::{Config, Features, Linear};

use ndarray::Ix2;

fn tracing() {
    use tracing::Level;
    use tracing_subscriber::fmt::time;

    tracing_subscriber::fmt()
        .compact()
        .with_ansi(true)
        .with_level(false)
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .with_timer(time::uptime())
        .init();
}
fn main() -> Result<()> {
    tracing();
    tracing::info!("Starting linear model example");

    let (sample, inputs, outputs) = (20, 5, 3);
    let features = Features::new(inputs, outputs);
    let config = Config::new("example", features.clone());
    let data = linarr::<f64, Ix2>(features).unwrap();

    let model: Linear<f64> = Linear::std(config).uniform();

    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
    println!("Predictions: {:?}", y);
    Ok(())
}
