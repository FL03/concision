/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused)]
#![cfg(test)]

extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::prelude::{linarr, Predict};
use linear::{Config, Features, Linear};

use lazy_static::lazy_static;
use ndarray::*;

const SAMPLES: usize = 20;
const INPUTS: usize = 5;
const OUTPUT: usize = 3;

lazy_static! {
    static ref FEATURES: Features = Features::new(INPUTS, OUTPUT);
    static ref CONFIG: Config = Config::new("test", FEATURES.clone());
    static ref SHAPE: (usize, usize, usize) = (SAMPLES, INPUTS, OUTPUT);
}

#[test]
fn test_linear() {
    let (samples, input, output) = SHAPE.clone();
    let features = FEATURES.clone();
    let config = Config::new("test", features);
    let model: Linear<f64> = Linear::std(config).init();
    let data = linarr::<f64, Ix2>(features).unwrap();
    let output = model.predict(&data).unwrap();
}
