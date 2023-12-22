#[cfg(test)]
extern crate concision_core;

use concision_core::prelude::{linarr, now, tril};
use ndarray::prelude::{array, Array2};

#[test]
fn test_tril() {
    let a = linarr::<f64, ndarray::Ix2>((3, 3)).unwrap();
    let b = array![[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]];
    assert_eq!(b, tril(&a));
}

#[test]
fn test_linarr() {
    let args: Array2<f64> = linarr((2, 3)).unwrap();
    assert_eq!(&args, &array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
}
