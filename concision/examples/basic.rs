/*
    Appellation: basic <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::prelude::ModelFeatures;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("Welcome to concision!");

    let features = ModelFeatures::default();
    println!("Features: {:?}", features);

    Ok(())
}

pub fn sample_padding(samples: usize) -> anyhow::Result<()> {
    use cnc::ops::pad;
    let arr = Array::range(0.0, (samples * samples) as f64, 1.0);
    let arr = arr.to_shape((samples, samples))?;
    let padded = pad(&arr, &[[0, 8]], 0.0.into());
    println!("{:?}", &padded);
    Ok(())
}
