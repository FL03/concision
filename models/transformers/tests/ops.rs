/*
    Appellation: ops <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_transformers as transformers;

use concision::linarr;
use ndarray::prelude::*;
use transformers::ops::*;

#[test]
fn test_merge() {
    let shape = (3, 4, 5);
    let dout = (4, 15);
    let arr = linarr::<f64, Ix3>(shape.clone()).unwrap();
    let a = arr.clone().merge().unwrap();
    let b = merge(&arr, 0, 1).unwrap();

    assert_eq!(a.dim(), dout);
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a, b);
}

#[test]
fn test_merge_batch() {
    let shape = (2, 3, 4, 5);
    let dout = (2, 4, 15);
    let arr = linarr::<f64, Ix4>(shape).unwrap();
    let a = arr.merge().unwrap();
    let b = merge(&arr, 1, 2).unwrap();

    assert_eq!(a.dim(), dout);
    assert_eq!(a, b);
}

#[test]
fn reshape_ops() {
    let dim_input: [usize; 3] = [2, 4, 6]; // (batch, seq, model)
    let dim_split = [2, 2, 4, 3]; // (batch, heads, seq, model)
    let data = linarr::<f64, Ix3>(dim_input).unwrap();

    let a = split_batch(&data, 2).unwrap();
    let b = a.merge().unwrap(); // merge_batch(&a).unwrap();

    assert_eq!(a.shape(), &dim_split);
    assert_eq!(b.shape(), &dim_input);
    assert_eq!(a, data.split(2).unwrap());
    for (i, &j) in b.indexed_iter() {
        assert_eq!(j, data[i]);
    }
}
