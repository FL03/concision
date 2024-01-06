extern crate concision;

use concision as cnc;

use cnc::core::ops::fft::*;
use cnc::prelude::{Arange, AsComplex, BoxResult};

use ndarray::prelude::*;

fn main() -> BoxResult {
    println!("Welcome to concision!");
    let samples = 8;

    let arr = Array1::<f64>::arange(samples).mapv(AsComplex::as_re);
    let buff = arr.clone().into_raw_vec();
    let plan = FftPlan::new(samples);
    println!("Permutations: {:?}", plan.plan());
    let res = ifft(buff.as_slice(), &plan);
    println!("{:?}", &res);
    Ok(())
}
