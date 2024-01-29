#[cfg(test)]
extern crate concision_s4;
use concision_s4 as s4;

use s4::prelude::{scan_ssm, scanner, SSMStore};

use lazy_static::lazy_static;
use ndarray::prelude::*;
use num::Complex;

const FEATURES: usize = 3;
const SAMPLES: usize = 16;

lazy_static! {
    static ref U: Array2<f64> = Array::range(0.0, FEATURES as f64, 1.0).insert_axis(Axis(1));
    static ref X0: Array1<f64> = Array1::zeros(FEATURES);
    static ref X1: Array1<Complex<f64>> = Array1::zeros(FEATURES);
    static ref A: Array2<f64> = Array::range(0.0, (FEATURES * FEATURES) as f64, 1.0)
        .into_shape((FEATURES, FEATURES))
        .unwrap();
    static ref B: Array2<f64> = Array::range(0.0, FEATURES as f64, 1.0).insert_axis(Axis(1));
    static ref C: Array2<f64> = Array::range(0.0, FEATURES as f64, 1.0).insert_axis(Axis(0));
}

#[test]
fn test_scan() {
    let u = U.clone();
    let x0 = X0.clone();

    let ssm = SSMStore::<f64>::from_features(FEATURES);
    let (a, b, c, _d) = ssm.clone().into();
    let scan1 = scanner(&A, &B, &C, &u, &x0);
    let scan2 = scan_ssm(&A, &B, &C, &u, &x0).expect("");
    println!("{:?}", &scan2);
    // let scan2 = ssm.scan(&u, &x0).unwrap();

    assert_eq!(&scan1, &scan2);
    assert!(false)
}
