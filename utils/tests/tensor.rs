extern crate concision_utils as utils;

use ndarray::prelude::*;
use utils::linarr;

#[test]
fn test_linarr() {
    let shape = (2, 3);
    let n = shape.0 * shape.1;
    let args = linarr::<f64, Ix2>(shape.clone()).unwrap();
    let exp = Array::linspace(0f64, (n - 1) as f64, n)
        .into_shape_clone(shape)
        .unwrap();
    assert_eq!(args, exp);
}

#[test]
fn test_tril() {
    let a = linarr::<f64, Ix2>((3, 3)).unwrap();
    let exp = array![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [6.0, 7.0, 8.0,]];
    assert_eq!(exp, utils::tril(&a));
}
