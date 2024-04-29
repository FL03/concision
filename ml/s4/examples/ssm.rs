// use concision_core as cnc;
extern crate concision_neural as neural;
extern crate concision_s4 as s4;

use neural::prelude::Predict;
use s4::ssm::{SSMConfig, SSMLayer};

use ndarray::prelude::*;

const FEATURES: usize = 4;
const SAMPLES: usize = 100;

fn main() -> anyhow::Result<()> {
    let u = Array::linspace(0.0, 1.0, SAMPLES);

    let config = SSMConfig::new(true, FEATURES, SAMPLES);
    let model = SSMLayer::<f64>::new(config).init()?;

    let res = model.predict(&u)?;
    println!("{:?}", res);

    Ok(())
}
