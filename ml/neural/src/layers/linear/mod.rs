/*
   Appellation: linear <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Layer
pub use self::{layer::*, utils::*};

pub(crate) mod layer;

use crate::params::{Biased, Weighted};
use ndarray::ScalarOperand;
use ndarray::prelude::Array2;
use num::Float;

pub trait LinearTransformation<T>
where
    T: Float,
{
    fn linear(&self, data: &Array2<T>) -> Array2<T>;
}

impl<S, T> LinearTransformation<T> for S
where
    S: Biased<T> + Weighted<T>,
    T: Float + ScalarOperand + 'static,
{
    fn linear(&self, data: &Array2<T>) -> Array2<T> {
        data.dot(&self.weights().t()) + self.bias()
    }
}

pub(crate) mod utils {
    use ndarray::prelude::{Array1, Array2};
    use num::Float;

    pub fn linear_transformation<T: Float + 'static>(
        data: &Array2<T>,
        bias: &Array1<T>,
        weights: &Array2<T>,
    ) -> Array2<T> {
        data.dot(&weights.t()) + bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let (inputs, outputs) = (2, 2);
        let data = Array2::<f64>::ones((inputs, outputs));
        let layer = LinearLayer::new(inputs, outputs);
        let linear = layer.linear(&data);
        assert_eq!(linear.dim(), (inputs, outputs));
    }
}
