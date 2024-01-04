// use concision_core as cnc;
use concision_s4 as s4;

use s4::randcomplex;

use ndarray::prelude::Ix2;
use num::complex::Complex;

fn main() -> anyhow::Result<()> {
    let c = randcomplex::<f64, Ix2>([2, 2]);

    println!("{:?}", &c);

    Ok(())
}
