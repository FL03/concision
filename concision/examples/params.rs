/*
    Appellation: params <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::init::NdRandom;
use cnc::params::Params;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_timer(tracing_subscriber::fmt::time::Uptime::default())
        .init();
    tracing::info!("Initialize some params...");
    let (m, n) = (8, 9);

    let inputs = Array1::linspace(0.0, 1.0, m);
    // initialize a 2-dimensional parameter set with 8 samples and 9 features
    let params = Params::<f64>::glorot_normal((m, n));
    tracing::info!("Initial Values: {params:?}");
    // validate the shape of the parameters
    assert_eq!(params.weights().shape(), &[m, n]);
    assert_eq!(params.bias().shape(), &[n]);
    // perform a forward pass through the parameters
    let y = params.forward(&inputs);
    assert_eq!(y.shape(), &[n]);
    tracing::info!("Forward pass: {y:?}");

    Ok(())
}
