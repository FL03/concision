/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, gradient::*, modes::*, utils::*};

pub(crate) mod descent;
pub(crate) mod gradient;
pub(crate) mod modes;

pub mod adam;
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
    use crate::core::prelude::BoxResult;
    use crate::neural::func::activate::Gradient;
    use crate::neural::models::exp::Module;
    use crate::neural::prelude::{Forward, ForwardIter};
    use ndarray::linalg::Dot;
    use ndarray::prelude::{Array, Array2, Dimension, NdFloat};
    use ndarray_stats::DeviationExt;
    use num::{FromPrimitive, Signed};
    use std::ops::Sub;

    pub fn gradient_descent<M, I, T, D>(
        _gamma: T,
        model: &mut M,
        _objective: impl Gradient<T, D>,
        data: &Array2<T>,
        targets: &Array<T, D>,
    ) -> anyhow::Result<f64>
    where
        D: Dimension,
        M: Clone + ForwardIter<Array2<T>, I, Output = Array<T, D>>,
        I: Forward<Array2<T>, Output = Array<T, D>>,
        T: FromPrimitive + NdFloat + Signed,
        Array2<T>: Dot<Array<T, D>, Output = Array<T, D>>,
    {
        let loss = model.forward(data).mean_sq_err(targets)?;
        Ok(loss)
    }

    pub fn gradient<'a, T, D, A>(
        gamma: T,
        model: &mut A,
        data: &Array2<T>,
        targets: &Array<T, D>,
        grad: impl Gradient<T, D>,
    ) -> BoxResult<f64>
    where
        A: Module<T, Output = Array<T, D>>,
        D: Dimension + 'a,
        T: FromPrimitive + NdFloat + Signed,
        Array2<T>: Dot<Array<T, D>, Output = Array<T, D>>,
        &'a Array2<T>: Sub<&'a Array<T, D>, Output = Array<T, D>>,
    {
        let (_samples, _inputs) = data.dim();
        let pred = model.predict(data)?;

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
        for p in model.parameters_mut().values_mut() {
            p.scaled_add(-gamma, &dw.t());
        }

        let loss = targets
            .mean_sq_err(&model.predict(data)?)
            .expect("Error when calculating the MSE of the model");
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::prelude::linarr;
    use crate::neural::func::activate::{LinearActivation, Sigmoid};
    use crate::neural::models::ModelParams;
    use crate::neural::prelude::{Features, Forward, Layer, LayerShape};
    use ndarray::prelude::{Array1, Ix2};

    #[test]
    fn test_gradient_descent() {
        let (epochs, gamma) = (10, 0.001);
        let (samples, inputs) = (20, 5);
        let outputs = 4;

        let _shape = (samples, inputs);

        let features = LayerShape::new(inputs, outputs);

        let x = linarr::<f64, Ix2>((samples, features.inputs())).unwrap();
        let y = linarr::<f64, Ix2>((samples, features.outputs())).unwrap();

        let mut shapes = vec![features];
        shapes.extend((0..3).map(|_| LayerShape::new(features.outputs(), features.outputs())));

        let mut model = ModelParams::<f64>::from_iter(shapes).init(true);

        let mut losses = Array1::zeros(epochs);
        for e in 0..epochs {
            let cost = gradient_descent(gamma, &mut model, Sigmoid, &x, &y)
                .expect("Gradient Descent Error");
            losses[e] = cost;
        }
        assert_eq!(losses.len(), epochs);
    }

    #[test]
    fn test_gradient() {
        let (samples, inputs, outputs) = (20, 5, 1);

        let (_epochs, _gamma) = (10, 0.001);

        let features = LayerShape::new(inputs, outputs);

        // Generate some example data
        let x = linarr::<f64, Ix2>((samples, features.inputs())).unwrap();
        let _y = linarr::<f64, Ix2>((samples, features.outputs())).unwrap();

        let model = Layer::<f64, LinearActivation>::from(features).init(true);

        let _pred = model.forward(&x);

        // let mut losses = Array1::zeros(epochs);
        // for e in 0..epochs {
        //     let cost = gradient(gamma, &mut model, &x, &y, Sigmoid).unwrap();
        //     losses[e] = cost;
        // }
        // assert_eq!(losses.len(), epochs);
        // assert!(losses.first() > losses.last());
    }
}
