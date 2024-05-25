#![allow(unused_imports)]
extern crate concision_core as concision;

use concision::nn::DropoutLayer;
use concision::Forward;
use ndarray::prelude::*;

#[test]
#[cfg(feature = "rand")]
fn test_dropout() {
    let shape = (512, 2048);
    let arr = Array2::<f64>::ones(shape);
    let dropout = DropoutLayer::new(0.5);
    let out = dropout.forward(&arr);

    assert!(arr.iter().all(|&x| x == 1.0));
    assert!(out.iter().any(|&x| x == 0.0));
}
