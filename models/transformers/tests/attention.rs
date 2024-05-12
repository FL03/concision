/*
    Appellation: attention <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_transformers as transformers;

use concision::{linarr, Matmul};
use transformers::{AttentionHead, Params};

use ndarray::prelude::*;

#[test]
fn test_qkv() {
    let shape = (2048, 10);
    let params = Params::<f64>::new(shape);
    assert_eq!(params.q(), &Array::default(shape));
}

#[test]
fn test_qkv_matmul() {
    let shape = (2048, 10);
    // generate some sample data
    let data = linarr(shape).unwrap();
    // initialize the parameters
    let params = Params::<f64>::ones(shape);
    // calculate the expected result
    let exp = Array2::<f64>::ones(shape).dot(&data.t());
    // calculate the result
    let res = params.matmul(&data.t());
    // compare the results
    assert_eq!(res.q(), &exp);
    assert_eq!(res.k(), &exp);
    assert_eq!(res.v(), &exp);
}

#[test]
fn test_attention_head() {
    let shape = (30, 3);

    let head = AttentionHead::<f64>::ones(shape);
    assert_eq!(head.q(), &Array::ones(shape));
}
