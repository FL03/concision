/*
    Appellation: basic <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;
extern crate ndarray as nd;

use cnc::ops::pad;
use nd::prelude::Array;

fn main() -> anyhow::Result<()> {
    println!("Welcome to concision!");
    let _samples = 8;

    let a = core::any::TypeId::of::<Biased>();
    let b = core::any::TypeId::of::<Unbiased>();
    assert_ne!(a, b);
    println!("{:?}\n{:?}", a, b);
    Ok(())
}
pub enum Biased {}
pub enum Unbiased {}

pub fn sample_padding(samples: usize) -> anyhow::Result<()> {
    let arr = Array::range(0.0, (samples * samples) as f64, 1.0).into_shape((samples, samples))?;
    let padded = pad(&arr, &[[0, 8]], 0.0.into());
    println!("{:?}", &padded);
    Ok(())
}
