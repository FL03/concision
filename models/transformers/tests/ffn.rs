/*
    Appellation: ffn <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;
extern crate concision_transformer as transformer;

use concision::prelude::{linarr, Predict};
use linear::Biased;
use transformer::ffn::FeedForwardNetwork;

use ndarray::prelude::*;

#[test]
fn test_ffn() {
    let (samples, d_model, d_ff) = (100, 30, 3);
    let model = FeedForwardNetwork::<f64, Ix2, Biased>::new(d_model, d_ff, Some(0.1));

    let data = linarr::<f64, Ix2>((samples, d_model)).unwrap();

    let pred = model.predict(&data).unwrap();
    assert_eq!(pred.dim(), (samples, d_model));
}
