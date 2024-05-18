/*
   Appellation: random <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as cnc;

use cnc::init::InitializeExt;
use ndarray::prelude::*;

#[test]
fn test_stdnorm() {
    let shape = [3, 3];
    let seed = 0u64;
    let a = Array2::<f64>::stdnorm(shape);
    let b = Array2::<f64>::stdnorm_from_seed(shape, seed);

    assert_eq!(a.shape(), shape);
    assert_eq!(a.shape(), b.shape());
}
