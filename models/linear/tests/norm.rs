/*
    Appellation: norm <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::{linarr, Forward};
use linear::{Biased, LayerNorm};
use ndarray::prelude::*;

#[test]
fn test_layer_norm() {
    let shape = (3, 3);
    let x = linarr::<f64, Ix2>(shape).unwrap() + 1f64;

    let ln = LayerNorm::<f64, Biased>::ones(shape);
    let y = ln.forward(&x);

    assert_eq!(y.dim(), shape);
}
