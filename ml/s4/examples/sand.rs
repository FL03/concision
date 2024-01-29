// use concision_core as cnc;
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    let (features, samples) = (4, 16);

    let u = Array::range(0.0, features as f64, 1.0).insert_axis(Axis(1));
    let x0 = Array1::<f64>::zeros(features);

    // let step = | st, u | {
    //     let x1 = st;
    //     let yk = u;
    //     Some(x1)
    // };
    // println!("{:?}", scan(step, u, x0.to_vec()));

    Ok(())
}
