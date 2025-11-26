/*
   Appellation: traits <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision_traits::*;
use ndarray::{Array2, Ix2, array};

#[test]
fn test_affine() {
    let x = array![[0.0, 1.0], [2.0, 3.0]];

    let y = x.affine(4.0, -2.0);
    assert_eq!(y, array![[-2.0, 2.0], [6.0, 10.0]]);
}

#[test]
fn test_inverse() {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[1.0, 2.0, 3.0,], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let exp = array![[-2.0, 1.0], [1.5, -0.5]];
    assert_eq!(Some(exp), a.inverse());
    assert_eq!(None, b.inverse());
}

#[test]
fn test_masked_fill() {
    let shape = (2, 2);
    let mask = array![[true, false], [false, true]];
    let arr = Array2::<f64>::from_shape_fn(shape, |(i, j)| (i * shape.1 + j) as f64);
    let a = arr.masked_fill(&mask, 0.0);
    assert_eq!(a, array![[0.0, 1.0], [2.0, 0.0]]);
}

#[test]
fn test_matrix_power() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(x.matpow(0), Array2::<f64>::eye(2));
    assert_eq!(x.matpow(1), x);
    assert_eq!(x.matpow(2), x.dot(&x));
}

#[test]
fn test_unsqueeze() {
    let arr = array![1, 2, 3, 4];
    let a = arr.clone().unsqueeze(0);
    assert_eq!(a.dim(), (1, 4));
    let b = arr.unsqueeze(1);
    assert_eq!(b.dim(), (4, 1));
}
