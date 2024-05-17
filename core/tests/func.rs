#![allow(unused_imports)]
extern crate concision_core as concision;

use concision::func::Dropout;
use concision::Forward;
use ndarray::prelude::*;

#[test]
#[cfg(feature = "rand")]
fn test_dropout() {
    let arr = Array2::<f64>::ones((2, 2));
    assert!(arr.iter().all(|&x| x == 1.0));
    let dropout = Dropout::new(0.5);
    let res = dropout.forward(&arr);
    assert!(res.iter().any(|&x| x == 0.0));
}
