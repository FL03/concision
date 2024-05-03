/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::{linarr, Predict};
use linear::{Config, Features, Linear};

use lazy_static::lazy_static;
use ndarray::*;

const SAMPLES: usize = 20;
const INPUTS: usize = 5;
const OUTPUT: usize = 3;

lazy_static! {
    static ref FEATURES: Features = Features::new(INPUTS, OUTPUT);
    static ref CONFIG: Config = Config::new("test", FEATURES.clone());
    static ref SAMPLE_DATA: Array<f64, Ix2> = linarr::<f64, Ix2>(FEATURES.clone()).unwrap();
    static ref SHAPE: (usize, usize, usize) = (SAMPLES, INPUTS, OUTPUT);
}

#[test]
fn test_linear() {
    let data = SAMPLE_DATA.clone();

    let model: Linear<f64> = Linear::std(CONFIG.clone()).uniform();

    let y = model.predict(&data).unwrap();
}
