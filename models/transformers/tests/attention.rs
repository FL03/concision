/*
    Appellation: attention <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_transformers as transformers;

use concision::{linarr, Matmul};
use transformers::{AttentionHead, QKV};

use ndarray::prelude::*;

#[test]
fn test_attention_params() {
    let shape = (2048, 10);
    let params = QKV::<f64>::new(shape);
    assert_eq!(params.q(), &Array::default(shape));

    let data: Array2<f64> = linarr(shape).unwrap();
    let exp = data.dot(&Array2::<f64>::ones(shape));
    let params = QKV::<f64>::ones(shape);
    let res = params.matmul(&data);
    assert_eq!(res.q(), &exp);
}

#[test]
fn test_attention_head() {
    let shape = (30, 3);
    
    let head = AttentionHead::<f64>::ones(shape);
    assert_eq!(head.q(), &Array::ones(shape));
}