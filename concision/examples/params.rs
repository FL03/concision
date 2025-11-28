/*
    Appellation: params <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::init::InitRand;
use cnc::params::Params;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .with_ansi(true)
        .init();
    tracing::info!("Initializing basic example...");
    let (m, n) = (8, 9);

    let inputs = Array1::linspace(0.0, 1.0, m);
    // initialize a 2-dimensional parameter set with 8 samples and 9 features
    let mut params = Params::<f64>::default((m, n));
    tracing::info!("Initial Values: {params:?}");
    // validate the shape of the parameters
    assert_eq!(params.weights().shape(), &[m, n]);
    assert_eq!(params.bias().shape(), &[n]);
    // initialize the parameters with random values
    params.assign_weights(&Array2::glorot_normal((m, n)));
    params.assign_bias(&Array1::glorot_normal((n,)));
    // validate the shape of the parameters
    assert_eq!(params.weights().shape(), &[m, n]);
    assert_eq!(params.bias().shape(), &[n]);
    tracing::info!("Randomized parameters: {params:?}");

    let y = params.forward(&inputs).expect("forward pass failed");
    assert_eq!(y.shape(), &[n]);
    tracing::info!("Forward pass: {y:?}");

    Ok(())
}
