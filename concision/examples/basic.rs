/*
    Appellation: basic <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![feature(autodiff)]

extern crate concision as cnc;
extern crate ndarray as nd;

use cnc::ops::pad;
use nd::prelude::Array;

fn add(x: f64, y: f64) -> f64 {
    x + y
}

fn main() -> anyhow::Result<()> {
    println!("Welcome to concision!");
    dbg!(add(1.0, 2.0));
    dbg!(add(3.0, 4.0));
    Ok(())
}
pub enum Biased {}
pub enum Unbiased {}

pub fn sample_padding(samples: usize) -> anyhow::Result<()> {
    let arr = Array::range(0.0, (samples * samples) as f64, 1.0);
    let arr = arr.to_shape((samples, samples))?;
    let padded = pad(&arr, &[[0, 8]], 0.0.into());
    println!("{:?}", &padded);
    Ok(())
}
