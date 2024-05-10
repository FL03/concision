/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
// #[allow(dead_code)]
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::prelude::{linarr, Sigmoid};
use linear::{Config, Features, Linear, Unbiased};

use lazy_static::lazy_static;
use ndarray::*;

const SAMPLES: usize = 20;
const D_MODEL: usize = 5;
const OUTPUTS: usize = 3;
const SHAPE: (usize, (usize, usize)) = (SAMPLES, (OUTPUTS, D_MODEL));

lazy_static! {
    static ref FEATURES: Features = Features::new(OUTPUTS, D_MODEL);
}

#[test]
fn test_config() {
    let dim = FEATURES.clone().into_dimension();
    let config = Config::from_dim(dim).biased();
    assert!(config.is_biased());
    let config = Config::from_dim(dim).unbiased();
    assert!(!config.is_biased());
}

#[test]
fn test_linear() {
    let (samples, (outputs, inputs)) = SHAPE;

    let model: Linear<f64> = Linear::from_features(inputs, outputs).uniform();

    let data = linarr::<f64, Ix2>((samples, inputs)).unwrap();
    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();

    assert_eq!(y.shape(), &[samples, outputs]);
}

#[test]
fn test_bias_ty() {
    use linear::{Biased, Unbiased};
    let (_samples, (outputs, inputs)) = SHAPE;

    let model: Linear<f64, Ix2, Biased> = Linear::from_features(inputs, outputs).uniform();
    assert!(model.is_biased());

    let model: Linear<f64, Ix2, Unbiased> = Linear::from_features(inputs, outputs).uniform();
    assert!(!model.is_biased());
}
