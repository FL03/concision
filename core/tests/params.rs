/*
   Appellation: params <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;

use concision::linarr;
use concision::params::{Parameter, ParamKind};
use ndarray::*;
use ndarray::linalg::Dot;

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
        (ParamKind::Bias, "bias"),
        (ParamKind::Weight, "weight"),
        (other.clone(), "test"),
        (ParamKind::other("mask"), "mask"),
    ];

    for (kind, expected) in &data {
        assert_eq!(kind.to_string(), expected.to_string());
    }
    
}
