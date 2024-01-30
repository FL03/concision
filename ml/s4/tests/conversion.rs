#[cfg(test)]
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;
use s4::ops::scan_ssm;

use core::prelude::{assert_atol, randc_normal};
use s4::cmp::kernel::kernel_dplr;
use s4::hippo::dplr::DPLR;
use s4::prelude::{casual_conv1d, discretize_dplr, k_conv, DPLRParams};

use ndarray::prelude::*;
use ndarray_linalg::flatten;
use num::complex::Complex;

const EPSILON: f64 = 1e-4;
const FEATURES: usize = 8;
const RNGKEY: u64 = 1;
const SAMPLES: usize = 16;

#[test]
fn test_conversion() {
    let step = (SAMPLES as f64).recip();
    // generate a random C matrix
    let c = randc_normal(RNGKEY, FEATURES);
    // Initialize a new DPLR Matrix
    let dplr = DPLR::<f64>::new(FEATURES);
    let (lambda, p, b, _) = dplr.clone().into();

    // CNN Form
    let kernel = {
        let params = DPLRParams::new(lambda.clone(), p.clone(), p.clone(), b.clone(), c.clone());
        kernel_dplr::<f64>(&params, step, SAMPLES)
    };
    // RNN Form
    let discrete = discretize_dplr(&lambda, &p, &p, &b, &c, step, SAMPLES).unwrap();
    let (ab, bb, cb) = discrete.into();

    let k2 = k_conv(&ab, &bb, &cb, SAMPLES);
    let k2r = k2.view().split_complex().re.to_owned();

    assert_atol(&kernel, &k2r, EPSILON);

    let u = Array::range(0.0, SAMPLES as f64, 1.0);
    let u2 = u.mapv(|i| Complex::new(i, 0.0)).insert_axis(Axis(1));
    // Apply the CNN
    let y1 = casual_conv1d(&u, &kernel).unwrap();

    // Apply the RNN
    let x0 = Array::zeros(FEATURES);
    let y2 = scan_ssm(&ab, &bb, &cb, &u2, &x0).unwrap();
    let y2r = {
        let tmp = y2.view().split_complex();
        flatten(tmp.re.to_owned())
    };

    assert_atol(&y1, &y2r, EPSILON)
}
