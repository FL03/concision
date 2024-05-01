/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused)]
#![cfg(test)]

extern crate concision_core as concision;
extern crate concision_linear as linear;
extern crate concision_neural as neural;

use concision::prelude::{linarr, Forward};
use linear::Features;

use lazy_static::lazy_static;

const SAMPLES: usize = 20;
const INPUTS: usize = 5;
const OUTPUT: usize = 3;

lazy_static! {
    static ref FEATURES: Features = Features::new(INPUTS, OUTPUT);
    static ref CONFIG: (usize, usize, usize) = (SAMPLES, INPUTS, OUTPUT);
}

#[test]
fn test_linear() {
    let (samples, input, output) = CONFIG.clone();
    let features = FEATURES.clone();
}
