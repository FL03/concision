/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]

#[cfg(not(feature = "std"))]
extern crate alloc;

extern crate concision_core as concision;

use concision::linarr;
use concision::params::{ParamKind, Parameter};
use ndarray::linalg::Dot;
use ndarray::prelude::{Ix1, Ix2};

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as Map;
#[cfg(feature = "std")]
use std::collections::HashMap as Map;

#[test]
fn test_parameter() {
    let a = linarr::<f64, Ix1>((3,)).unwrap();
    let p = linarr::<f64, Ix2>((3, 3)).unwrap();
    let mut param = Parameter::<f64, Ix2>::new((10, 1), ParamKind::Bias, "bias");
    param.set_params(p.clone());

    assert_eq!(param.kind(), &ParamKind::Bias);
    assert_eq!(param.name(), "bias");
    assert_eq!(param.dot(&a), p.dot(&a));
}

#[test]
fn test_param_kind_map() {
    let name = "test";
    let other = ParamKind::other(name);

    let data = [
        (ParamKind::Bias, 0),
        (ParamKind::Weight, 1),
        (other.clone(), 2),
        (ParamKind::other("mask"), 3),
    ];
    let store = Map::<ParamKind, usize>::from_iter(data);
    assert_eq!(store.get(&ParamKind::Bias), Some(&0));
    assert_eq!(store.get(&other), Some(&2));
}
