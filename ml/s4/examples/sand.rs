// use concision_core as cnc;
extern crate concision_s4;

// use concision_core as core;
// use concision_s4 as s4;

use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    let (_features, _samples) = (4, 16);

    let _u = Array::range(0.0, _samples as f64, 1.0);

    Ok(())
}
