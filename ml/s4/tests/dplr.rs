#[cfg(test)]
extern crate concision_core;
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;

use ndarray::prelude::*;
use ndarray_linalg::flatten;
use num::complex::ComplexFloat;

use core::prelude::{AsComplex, Conjugate, GenerateRandom, Power};
use s4::cmp::{kernel_dplr, DPLRParams};
use s4::hippo::dplr::DPLR;
use s4::ops::{discretize, k_convolve};
use s4::prelude::randcomplex;

#[test]
fn test_gen_dplr() {
    let (features, samples) = (4, 16);

    let eye = Array2::<f64>::eye(features);

    let step = (samples as f64).recip();

    let dplr = DPLR::<f64>::new(features);

    let b2 = dplr.b.clone().insert_axis(Axis(1));

    let p2 = dplr.p.clone().insert_axis(Axis(1));

    let a = Array::from_diag(&dplr.lambda) - p2.dot(&p2.conj().t());

    let c = randcomplex::<f64, Ix1>(features);
    let c2 = c.clone().insert_axis(Axis(0));

    let discrete = {
        let tmp = discretize(&a, &b2, &c2, step.as_re());
        assert!(tmp.is_ok(), "discretize failed: {:?}", tmp.err().unwrap());
        tmp.unwrap()
    };

    let (ab, bb, cb) = discrete.into();
    //
    let ak = k_convolve(&ab, &bb, &cb.conj(), samples);
    println!("Ak: {:?}", ak.shape());
    //
    let cc = (&eye - ab.pow(samples)).conj().t().dot(&flatten(cb));
    //
    let params = DPLRParams::new(
        dplr.lambda.clone(),
        dplr.p.clone(),
        dplr.p.clone(),
        dplr.b.clone(),
        cc,
    );
    //
    let kernal = kernel_dplr::<f64>(&params, step, samples);
    println!("Kernal: {:?}", kernal.shape());

    let a_real = ak.mapv(|i| i.re());
    let err = (&a_real - &kernal).mapv(|i| i.abs());
    assert!(
        err.mean().unwrap() <= 1e-4,
        "Error: {:?}\nTolerance: {:?}",
        err.mean().unwrap(),
        1e-4
    );
}

#[test]
fn test_discretize_dplr() {
    let (features, samples) = (8, 16);

    let step = (samples as f64).recip();

    let dplr = DPLR::<f64>::new(features);

    let c = Array1::<f64>::stdnorm(features);

    // let kernal = kernel_dplr(lambda, p, q, b, c, step, l)
}
