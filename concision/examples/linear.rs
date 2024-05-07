/*
    Appellation: linear <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::func::Sigmoid;
use cnc::linear::{Config, Features, Linear};
use cnc::{linarr, Predict, Result};
use ndarray::{IntoDimension, Ix2};

fn tracing() {
    use tracing::Level;
    use tracing_subscriber::fmt::time;

    tracing_subscriber::fmt()
        .compact()
        .with_ansi(true)
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .with_timer(time::uptime())
        .init();
}

fn main() -> Result<()> {
    tracing();
    tracing::info!("Starting linear model example");

    let (samples, dmodel, features) = (20, 5, 3);
    let shape = Features::new(features, dmodel);
    let config = Config::from_dim(shape.into_dimension()).biased();
    let data = linarr::<f64, Ix2>((samples, dmodel)).unwrap();

    let model: Linear<f64> = Linear::std(config).uniform();

    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
    assert_eq!(y.dim(), (samples, features));
    println!("Predictions: {:?}", y);

    Ok(())
}
