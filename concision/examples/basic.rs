extern crate concision;

use concision as cnc;

use cnc::core::ops::pad;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("Welcome to concision!");
    let samples = 8;

    let arr = Array::range(0.0, (samples * samples) as f64, 1.0).into_shape((samples, samples))?;
    let padded = pad(&arr, &[[0, 8]], 0.0.into());
    println!("{:?}", &padded);
    Ok(())
}
