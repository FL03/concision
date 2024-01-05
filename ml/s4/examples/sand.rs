// use concision_core as cnc;
extern crate concision_s4;

use concision_s4 as s4;

use s4::randcomplex;

use ndarray::prelude::Ix2;
use ndarray_linalg::Scalar;
use num::complex::Complex;

fn main() -> anyhow::Result<()> {
    let i = Complex::<f64>::i();
    println!("{:?}", i.add_real(1.0));
    let c = randcomplex::<f64, Ix2>([2, 2]);

    println!("{:?}", &c);

    Ok(())
}
