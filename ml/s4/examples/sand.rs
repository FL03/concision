// use concision_core as cnc;
extern crate concision_s4;

// use concision_core as core;
// use concision_s4 as s4;

use ndarray::prelude::*;
use ndarray_conv::{Conv2DFftExt, PaddingMode, PaddingSize};

const _FEATURES: usize = 4;
const SAMPLES: usize = 8;

fn main() -> anyhow::Result<()> {
    let u = Array::range(0.0, (SAMPLES * SAMPLES) as f64, 1.0).into_shape((SAMPLES, SAMPLES))?;
    let k = Array::range(0.0, (SAMPLES * SAMPLES) as f64, 1.0).into_shape((SAMPLES, SAMPLES))?;

    let size = PaddingSize::Full; // PaddingSize::Custom([0; SAMPLES], [0; SAMPLES]);
    let mode = PaddingMode::Const(0.0);
    let res = u.conv_2d_fft(&k, size, mode);
    println!("{:?}", res);

    Ok(())
}
