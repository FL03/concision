/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, gradient::*, modes::*, utils::*};

pub(crate) mod descent;
pub(crate) mod gradient;
pub(crate) mod modes;

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
    use crate::neural::func::activate::Gradient;
    use crate::neural::prelude::{Forward, ForwardIter, Parameterized, Params};
    use ndarray::linalg::Dot;
    use ndarray::prelude::{Array, Array1, Array2, Dimension, NdFloat};
    use ndarray_stats::DeviationExt;
    use num::{FromPrimitive, Signed};

    pub fn gradient_descent<M, T, D>(
        gamma: T,
        model: &mut M,
        objective: impl Gradient<T, D>,
    ) -> anyhow::Result<f64>
    where
        D: Dimension,
        M: Forward<Array2<T>, Output = Array<T, D>> + Parameterized<T, D>,
        T: FromPrimitive + NdFloat,
    {
        let loss = 0.0;
        Ok(loss)
    }

    pub fn gradient<T, D, A>(
        gamma: T,
        model: &mut A,
        data: &Array2<T>,
        targets: &Array<T, D>,
        grad: impl Gradient<T, D>,
    ) -> f64
    where
        A: Forward<Array2<T>, Output = Array<T, D>> + Parameterized<T, D>,
        D: Dimension,
        T: FromPrimitive + NdFloat + Signed,
        <A as Parameterized<T, D>>::Params: Params<T, D> + 'static,
        Array2<T>: Dot<Array<T, D>, Output = Array<T, D>>,
    {
        let (_samples, _inputs) = data.dim();
        let pred = model.forward(data);

        let ns = T::from(data.len()).unwrap();

        let errors = &pred - targets;
        // compute the gradient of the objective function w.r.t. the model's weights
        let dz = &errors * grad.gradient(&pred);
        // compute the gradient of the objective function w.r.t. the model's weights
        let dw = data.t().to_owned().dot(&dz) / ns;
        // let dw = - model.params().bias() * dz + data.t().to_owned().dot(&dz)  / ns;
        // compute the gradient of the objective function w.r.t. the model's bias
        // let db = dz.sum_axis(Axis(0)) / ns;
        // // Apply the gradients to the model's learnable parameters
        // model.params_mut().bias_mut().scaled_add(-gamma, &db.t());

        model.params_mut().weights_mut().scaled_add(-gamma, &dw.t());

        let loss = targets
            .mean_sq_err(&model.forward(data))
            .expect("Error when calculating the MSE of the model");
        loss
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::prelude::linarr;
    use crate::neural::func::activate::{Linear, Sigmoid};
    use crate::neural::prelude::{Features, Layer, LayerShape, Parameterized, Weighted};
    use ndarray::prelude::{Array, Array1, Dimension};
    use num::Float;

    #[test]
    fn test_gradient_descent() {
        let (_samples, inputs, outputs) = (20, 5, 1);

        let (epochs, gamma) = (10, 0.001);

        let features = LayerShape::new(inputs, outputs);

        let mut model = Layer::<f64, Linear>::from(features).init(true);

        let mut losses = Array1::zeros(epochs);
        for e in 0..epochs {
            let cost =
                gradient_descent(gamma, &mut model, Sigmoid).expect("Gradient Descent Error");
            losses[e] = cost;
        }
        assert_eq!(losses.len(), epochs);
    }

    #[test]
    fn test_gradient() {
        let (samples, inputs, outputs) = (20, 5, 1);

        let (epochs, gamma) = (10, 0.001);

        let features = LayerShape::new(inputs, outputs);

        // Generate some example data
        let x = linarr((samples, features.inputs())).unwrap();
        let y = linarr((samples, features.outputs())).unwrap();

        let mut model = Layer::<f64, Linear>::from(features).init(true);

        let mut losses = Array1::zeros(epochs);
        for e in 0..epochs {
            let cost = gradient(gamma, &mut model, &x, &y, Sigmoid);
            losses[e] = cost;
        }
        assert_eq!(losses.len(), epochs);
        assert!(losses.first() > losses.last());
    }
}
