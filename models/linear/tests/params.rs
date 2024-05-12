/*
    Appellation: params <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::Predict;
use linear::params::{LinearParams, Param, Unbiased};
use linear::Features;

use core::str::FromStr;
use ndarray::prelude::*;

const SAMPLES: usize = 20;
const INPUTS: usize = 5;
const DMODEL: usize = 3;

#[test]
fn test_keys() {
    for i in [(Param::Bias, "bias"), (Param::Weight, "weight")].iter() {
        let kind = Param::from_str(i.1).unwrap();
        assert_eq!(i.0, kind);
    }
}

#[test]
fn test_linear_params() {
    let (samples, inputs, outputs) = (SAMPLES, INPUTS, DMODEL);
    let features = Features::new(outputs, inputs);
    let data = Array2::<f64>::zeros((samples, inputs));
    let params = LinearParams::biased(features).uniform();
    let y: Array2<f64> = params.predict(&data).unwrap();
    assert_eq!(y.dim(), (samples, outputs));
}

#[test]
fn test_ndbuilders() {
    let shape = (300, 10);
    let params = LinearParams::<f64>::ones(shape);
    assert!(params.is_biased());
    let params = LinearParams::<usize, Unbiased>::zeros(shape);
    assert!(!params.is_biased());
}
