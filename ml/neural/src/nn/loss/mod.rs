/*
    Appellation: loss <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Loss Functions
//!
//! Loss functions consider the differences between predicted and target outputs.
//! Overall, neural network models aim to minimize the average loss by adjusting certain hyperparameters,
//! the weights and biases.

pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

pub mod regress;

pub trait Loss {
    fn loss(&self, pred: &[f64], target: &[f64]) -> f64;
}



pub(crate) mod utils {
    use ndarray::Array1;

    pub fn mae(pred: Array1<f64>, target: Array1<f64>) -> anyhow::Result<f64> {
        if pred.shape() != target.shape() {
            return Err(anyhow::anyhow!(
                "Mismatched shapes: {:?} and {:?}",
                pred.shape(),
                target.shape()
            ));
        }
        // the number of elements in the array
        let n = pred.len() as f64;
        let mut res = (target - pred).mapv(|x| x.abs()).sum();
        res /= n;
        Ok(res)
    }

    pub fn mse(pred: Array1<f64>, target: Array1<f64>) -> anyhow::Result<f64> {
        if pred.shape() != target.shape() {
            return Err(anyhow::anyhow!(
                "Mismatched shapes: {:?} and {:?}",
                pred.shape(),
                target.shape()
            ));
        }

        let n = pred.len() as f64;
        let mut res = (target - pred).mapv(|x| x.powi(2)).sum();
        res /= n;
        Ok(res)
    }
}
