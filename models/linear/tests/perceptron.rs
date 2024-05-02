/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused)]
#![cfg(test)]

extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::func::activate::softmax;
use concision::traits::Forward;
use linear::Perceptron;
use ndarray::*;

#[test]
fn perceptron() {
    let bias = 0.0;

    let data = array![[10.0, 10.0, 6.0, 1.0, 8.0]];
    let weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
    let neuron = Perceptron::<f64>::new(Box::new(softmax), 5).with_weights(weights.clone());

    let linear = data.dot(&weights) + bias;
    let exp = softmax(&linear);

    assert_eq!(exp, neuron.forward(&data));
}
