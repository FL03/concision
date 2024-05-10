/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[allow(dead_code)]
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::func::Sigmoid;
use concision::{linarr, Predict};
use linear::{Config, Features, Linear};

use lazy_static::lazy_static;
use ndarray::*;

const SAMPLES: usize = 20;
const D_MODEL: usize = 5;
const DOUT: usize = 3;

lazy_static! {
    static ref FEATURES: Features = Features::new(DOUT, D_MODEL);
    static ref CONFIG: Config =
        Config::from_dim(FEATURES.clone().into_dimension()).with_name("test_model");
    static ref SAMPLE_DATA: Array<f64, Ix2> = linarr::<f64, Ix2>((SAMPLES, D_MODEL)).unwrap();
    static ref SHAPE: (usize, (usize, usize)) = (SAMPLES, (DOUT, D_MODEL));
}

#[test]
fn test_linear() {
    let (samples, (outputs, inputs)) = *SHAPE;

    let model: Linear<f64> = Linear::from_config(CONFIG.clone()).uniform();

    let data = SAMPLE_DATA.clone();
    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();

    assert_eq!(y.shape(), &[samples, outputs]);
}
