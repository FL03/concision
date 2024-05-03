/*
    Appellation: params <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]

extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::func::activate::{softmax, Softmax};
use concision::Predict;
use linear::{Features, LinearParams};
use ndarray::*;

#[test]
fn test_linear_params() {
    let (samples, inputs, outputs) = (20, 5, 3);
    let features = Features::new(inputs, outputs);
    let data = Array2::<f64>::zeros((samples, inputs));
    let params = LinearParams::default(features.clone()).init_uniform(true);
    let y = params.forward(&data).unwrap();
    assert_eq!(y.dim(), (samples, outputs));
}
