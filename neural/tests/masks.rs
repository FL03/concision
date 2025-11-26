extern crate concision_core as cnc;
extern crate concision_neural as neural;

use cnc::Forward;
use concision_neural::error::NeuralError;
use ndarray::prelude::*;
use neural::Dropout;

#[test]
fn test_dropout() -> Result<(), NeuralError> {
    let shape = (512, 2048);
    let arr = Array2::<f64>::ones(shape);
    let dropout = Dropout::new(0.5);
    let out = dropout.forward(&arr).expect("Dropout forward pass failed");

    assert!(arr.iter().all(|&x| x == 1.0));
    assert!(out.iter().any(|x| x == &0f64));

    Ok(())
}
