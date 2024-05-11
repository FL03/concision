// use concision_core as cnc;
extern crate concision_neural as neural;
extern crate concision_s4 as s4;

use neural::prelude::Predict;
use s4::prelude::{S4Config, S4Layer};

use ndarray::prelude::*;

const FEATURES: usize = 4;
const SAMPLES: usize = 100;

fn main() -> anyhow::Result<()> {
    let u = Array::linspace(0.0, 1.0, SAMPLES);

    let config = S4Config::new(true, FEATURES, SAMPLES);
    let model = S4Layer::<f64>::new(config).init()?;

    let res = model.predict(&u)?;
    println!("{:?}", res);

    Ok(())
}
