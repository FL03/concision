#![allow(unused_imports)]

extern crate concision_core as cnc;
extern crate concision_neural as nn;

use nn::layers::attention::{Qkv, ScaledDotProductAttention};

use ndarray::prelude::*;

#[test]
fn test_attention() {
    let (m, n) = (7, 10);
    let qkv = Qkv::<f64>::ones((m, n));
    // initialize the scaled dot-product attention layer
    let layer = ScaledDotProductAttention::<f64>::new(0.1, 1.0);
    // compute the attention scores
    let z_score = layer.attention(&qkv);
    // verify the output dimensions
    assert_eq!(z_score.shape(), &[m, n]);
}
