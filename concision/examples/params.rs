/*
    Appellation: params <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

#[cfg(feature = "rand")]
use cnc::init::NdRandom;
use cnc::params::Params;

use ndarray::prelude::*;

fn main() -> cnc::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();
    tracing::info! { "Initialize some params..." }
    let (m, n) = (8, 9);

    let inputs = Array1::linspace(0.0, 1.0, m);
    // initialize a 2-dimensional parameter set with 8 samples and 9 features
    #[cfg(feature = "rand")]
    let params = Params::<f64>::glorot_normal((m, n));
    #[cfg(not(feature = "rand"))]
    let params = Params::<f64>::ones((m, n));
    // log the initial values of the parameters
    tracing::info! { "Initial Values: {params:?}" }
    // validate the shape of the parameters
    assert_eq! { params.weights().shape(), &[m, n] }
    assert_eq! { params.bias().shape(), &[n] }
    // perform a forward pass through the parameters
    let y = params.forward(&inputs);
    assert_eq! { y.shape(), &[n] }
    tracing::info! { "Forward pass: {y:?}" }

    Ok(())
}
