/*
   Appellation: like <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as cnc;
use cnc::traits::{Affine, AsComplex, Matpow};
use ndarray::prelude::*;
use num::Complex;

#[test]
fn test_affine() {
    let x = array![[0.0, 1.0], [2.0, 3.0]];

    let y = x.affine(4.0, -2.0);
    assert_eq!(y, array![[-2.0, 2.0], [6.0, 10.0]]);
}

#[test]
fn test_as_complex() {
    let x = 1.0;
    let y = x.as_re();
    assert_eq!(y, Complex::new(1.0, 0.0));
}

#[test]
fn test_matrix_power() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(x.pow(0), Array2::<f64>::eye(2));
    assert_eq!(x.pow(1), x);
    assert_eq!(x.pow(2), x.dot(&x));
}
