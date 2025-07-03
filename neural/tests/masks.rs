extern crate concision_core as cnc;
extern crate concision_neural as neural;

use cnc::Forward;
use concision_neural::error::ModelError;
use ndarray::prelude::*;
use neural::Dropout;

#[test]
fn test_dropout() -> Result<(), ModelError> {
    let shape = (512, 2048);
    let arr = Array2::<f64>::ones(shape);
    let dropout = Dropout::new(0.5);
    let out = dropout.forward(&arr)?;

    assert!(arr.iter().all(|&x| x == 1.0));
    assert!(out.iter().any(|x| x == &0f64));

    Ok(())
}
