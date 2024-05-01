/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]
extern crate concision_core as concision;
extern crate concision_linear as linear;
extern crate concision_neural as neural;

use concision::prelude::{linarr, Forward};
use linear::model::LinearLayer;
use linear::{LinearShape, Node};
use ndarray::prelude::Ix2;
use neural::prelude::Softmax;

#[test]
fn test_linear_layer() {
    let (samples, inputs, outputs) = (20, 5, 3);
    let features = LinearShape::new(inputs, outputs);

    let args = linarr::<f64, Ix2>((samples, inputs)).unwrap();

    let layer = LinearLayer::<f64, Softmax>::from(features).init(true);

    let pred = layer.forward(&args);

    assert_eq!(pred.dim(), (samples, outputs));

    let nodes = (0..outputs)
        .map(|_| Node::<f64>::new(inputs).init(true))
        .collect::<Vec<_>>();
    let layer = LinearLayer::<f64, Softmax>::from_iter(nodes);
    assert_eq!(layer.features(), &features);
}

#[test]
fn test_linear_iter() {
    let (_samples, inputs, outputs) = (20, 5, 3);
    let features = LinearShape::new(inputs, outputs);

    let layer = LinearLayer::<f64, Softmax>::from(features).init(true);

    for node in layer.into_iter() {
        assert!(node.is_biased());
        assert_eq!(node.features(), inputs);
        assert_eq!(node.bias().as_ref().unwrap().dim(), ());
    }
}
