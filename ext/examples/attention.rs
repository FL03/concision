/*
    Appellation: attention <example>
    Created At: 2025.11.28:13:41:41
    Contrib: @FL03
*/
extern crate concision as cnc;

use concision_ext::attention::{Qkv, ScaledDotProductAttention};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .without_time()
        .init();
    let (m, n) = (7, 10);
    let qkv = Qkv::<f64>::ones((m, n));
    // initialize the scaled dot-product attention layer
    let layer = ScaledDotProductAttention::<f64>::new(0.1, 1.0);
    // compute the attention scores
    let z_score = layer.attention(&qkv);
    println!("z_score: {:?}", z_score);

    Ok(())
}
