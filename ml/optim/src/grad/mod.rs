/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, gradient::*, utils::*};

pub(crate) mod descent;
pub(crate) mod gradient;

pub mod sgd;

pub struct BatchParams {
    pub batch_size: usize,
}

pub struct DescentParams {
    pub batch_size: usize,
    pub epochs: usize,
    pub gamma: f64, // learning rate

    pub lambda: f64, // decay rate
    pub mu: f64,     // momentum rate
    pub nesterov: bool,
    pub tau: f64, // momentum damper
}

pub(crate) mod utils {
    use ndarray::prelude::{Array, Array1, Dimension, NdFloat};
    use num::FromPrimitive;

    pub fn gradient_descent<T, D>(
        params: &mut Array<T, D>,
        epochs: usize,
        gamma: T,
        partial: impl Fn(&Array<T, D>) -> Array<T, D>,
    ) -> Array1<T> where D: Dimension, T: FromPrimitive + NdFloat {
        let mut losses = Array1::zeros(epochs);
        for e in 0..epochs {
            let grad = partial(params);
            params.scaled_add(-gamma, &grad);
            losses[e] = params.mean().unwrap_or_else(T::zero);
        }
        losses
    }

    pub fn gradient_descent_step<T, D>(
        params: &mut Array<T, D>,
        gamma: T,
        partial: impl Fn(&Array<T, D>) -> Array<T, D>,
    ) -> T where D: Dimension, T: FromPrimitive + NdFloat {
        let grad = partial(params);
        params.scaled_add(-gamma, &grad);
        params.mean().unwrap_or_else(T::zero)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::neural::prelude::{Features, Layer, LinearActivation, Weighted};
    use ndarray::prelude::Array2;

    fn test_grad(args: &Array2<f64>) -> Array2<f64> {
        args.clone()
    }

    #[test]
    fn descent()  {
        let (_samples, inputs) = (20, 5);
        let outputs = 1;

        let (epochs, gamma) = (10, 0.001);

        let features = Features::new(inputs, outputs);

        let mut model = Layer::<f64, LinearActivation>::from(features).init(true);

        let losses = gradient_descent(&mut model.weights_mut(), epochs, gamma, test_grad);
        assert_eq!(losses.len(), epochs);
    }
}
