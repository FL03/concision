/*
    Appellation: basic <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::init::Initialize;
use cnc::params::Params;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();
    tracing::info!("Initializing basic example...");
    let (m, n) = (8, 9);
    // initialize a 2-dimensional parameter set with 8 samples and 9 features
    let mut params = Params::<f64>::default((m, n));
    // validate the shape of the parameters
    assert_eq!(params.weights().shape(), &[m, n]);
    assert_eq!(params.bias().shape(), &[n]);
    // initialize the parameters with random values
    params.assign_weights(&Array2::glorot_normal((m, n), m, n));
    params.assign_bias(&Array1::glorot_normal((n,), n, m));

    Ok(())
}
