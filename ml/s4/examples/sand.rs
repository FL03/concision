// use concision_core as cnc;
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;

use core::prelude::{Arange, AsComplex};
use ndarray::prelude::*;
use rustfft::FftPlanner;
use s4::prelude::cauchy;
use s4::ssm::{SSMConfig, SSM};

fn main() -> anyhow::Result<()> {
    let (features, samples) = (4, 16);

    let arr = Array::<f64, Ix1>::arange(samples).mapv(AsComplex::as_re);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(samples);
    let mut buffer = arr.to_vec();
    println!("Buffer: {:?}", &buffer);
    fft.process(buffer.as_mut_slice());

    let res = Array::from_vec(buffer);

    println!("FFT:\n\n{:?}\n", res);

    let config = SSMConfig::new(true, features, samples);
    let _ssm = SSM::<f64>::create(config);

    let a = Array1::<f64>::arange(features * features).into_shape((features, features))? + 1.0;
    let b = Array1::<f64>::arange(features).insert_axis(Axis(1));
    let c = Array1::<f64>::arange(features).insert_axis(Axis(0)) * 2.0;
    println!("Cauchy:\n\n{:?}", cauchy(&a, &b, &c));

    Ok(())
}
