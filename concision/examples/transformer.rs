/*
    Appellation: transformer <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use approx::AbsDiffEq;
use cnc::transformer::AttentionHead;
use cnc::prelude::Result;
use ndarray::Array2;

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
    tracing::info!("Starting up the transformer model example...");

    let shape = (3, 3);
    let head = AttentionHead::<f64>::ones(shape);
    let score = head.attention();
    assert!(score.attention().abs_diff_eq(&Array2::from_elem(shape, 1f64/3f64), 1e-6));
    println!("{:?}", score);


    

    Ok(())
}
