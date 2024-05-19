/*
    Appellation: linear <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::linear::Features;
use cnc::prelude::{linarr, InitializeExt, Linear, Result, Sigmoid};
use ndarray::Ix2;

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
    let samples = 20;
    let (dm, dn) = (5, 3);
    let features = Features::new(dn, dm);
    let data = linarr::<f64, Ix2>((samples, dm)).unwrap();

    let model = Linear::<f64>::lecun_normal(features, dm);
    assert!(model.is_biased());

    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
    assert_eq!(y.dim(), (samples, dn));
    println!("Predictions:\n{:#?}", &y);

    Ok(())
}
