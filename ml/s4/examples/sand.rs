// use concision_core as cnc;
extern crate concision_neural as neural;
extern crate concision_s4 as s4;

use neural::prelude::Predict;
use s4::ssm::{SSMConfig, SSMLayer};

use ndarray::prelude::*;

const FEATURES: usize = 4;
const SAMPLES: usize = 100;

fn main() -> anyhow::Result<()> {
    let u = Array::range(0.0, (SAMPLES * SAMPLES) as f64, 1.0);

    let config = SSMConfig::new(true, FEATURES, SAMPLES);
    let model = SSMLayer::<f64>::create(config)?;

    let res = model.predict(&u)?;
    println!("{:?}", res);

    Ok(())
}
