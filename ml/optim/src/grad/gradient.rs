/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use ndarray::prelude::{Array1, Array2};

pub type BiasGradient<T = f64> = Array1<T>;

pub type WeightGradient<T = f64> = Array2<T>;

pub struct Gradients {
    pub bias: BiasGradient,
    pub weights: WeightGradient,
}

impl Gradients {
    pub fn new(bias: BiasGradient, weights: WeightGradient) -> Self {
        Self { bias, weights }
    }

    pub fn zeros(inputs: usize, outputs: usize) -> Self {
        Self {
            bias: Array1::zeros(outputs),
            weights: Array2::zeros((inputs, outputs)),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_gradient() {
        let (samples, inputs) = (20, 5);
        let _shape = (samples, inputs);
    }
}
