/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::func::activate::Sigmoid;
use crate::neural::models::ModelParams;
use crate::neural::prelude::{Forward, Gradient, Weighted};
use ndarray::prelude::{Array2, Axis, NdFloat};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};

pub struct Grad<T = f64, O = Sigmoid>
where
    O: Gradient<T>,
    T: Float,
{
    gamma: T,
    params: ModelParams<T>,
    objective: O,
}

impl<T, O> Grad<T, O>
where
    O: Gradient<T>,
    T: Float,
{
    pub fn new(gamma: T, params: ModelParams<T>, objective: O) -> Self {
        Self {
            gamma,
            params,
            objective,
        }
    }

    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut T {
        &mut self.gamma
    }

    pub fn objective(&self) -> &O {
        &self.objective
    }

    pub fn model(&self) -> &ModelParams<T> {
        &self.params
    }

    pub fn model_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }
}

impl<T, O> Grad<T, O>
where
    O: Gradient<T>,
    T: NdFloat + Signed,
{
    pub fn gradient(&mut self, data: &Array2<T>, targets: &Array2<T>) -> anyhow::Result<f64> {
        let lr = self.gamma();
        // the number of layers in the model
        let depth = self.model().len();
        // the gradients for each layer
        let mut grads = Vec::with_capacity(depth);
        // a store for the predictions of each layer
        let mut store = vec![data.clone()];
        // compute the predictions for each layer
        for layer in self.model().clone().into_iter() {
            let pred = layer.forward(&store.last().unwrap());
            store.push(pred);
        }
        // compute the error for the last layer
        let error = store.last().unwrap() - targets;
        // compute the error gradient for the last layer
        let dz = &error * self.objective.gradient(&error);
        // push the error gradient for the last layer
        grads.push(dz.clone());

        for i in (1..depth).rev() {
            // get the weights for the current layer
            let wt = self.params[i].weights().t();
            // compute the delta for the current layer w.r.t. the previous layer
            let dw = grads.last().unwrap().dot(&wt);
            // compute the gradient w.r.t. the current layer's predictions
            let dp = self.objective.gradient(&store[i]);
            // compute the gradient for the current layer
            let gradient = dw * &dp;
            grads.push(gradient);
        }
        // reverse the gradients so that they are in the correct order
        grads.reverse();
        // update the parameters for each layer
        for i in 0..depth {
            let grad = &grads[i];
            println!("Layer ({}) Gradient (dim): {:?}", i, grad.shape());
            let wg = &store[i].t().dot(grad);
            let _bg = grad.sum_axis(Axis(0));
            self.params[i].weights_mut().scaled_add(-lr, &wg.t());
        }
        let loss = self.model().forward(data).mean_sq_err(targets)?;
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::neural::models::ModelParams;
    use crate::neural::prelude::{Features, LayerShape, Sigmoid};

    use ndarray::prelude::Ix2;

    pub fn assert_ok<T, E>(result: Result<T, E>) -> T
    where
        E: std::fmt::Debug,
        T: std::fmt::Debug,
    {
        assert!(result.is_ok(), "{:?}", result);
        result.unwrap()
    }

    #[test]
    fn test_gradient() {
        let (samples, inputs) = (20, 5);
        let outputs = 4;

        let _shape = (samples, inputs);

        let features = LayerShape::new(inputs, outputs);

        let x = linarr::<f64, Ix2>((samples, features.inputs())).unwrap();
        let y = linarr::<f64, Ix2>((samples, features.outputs())).unwrap();

        let mut shapes = vec![features];
        shapes.extend((0..3).map(|_| LayerShape::new(features.outputs(), features.outputs())));

        let mut model = ModelParams::<f64>::from_iter(shapes).init(true);

        let mut grad = Grad::new(0.01, model.clone(), Sigmoid);

        let mut losses = Vec::new();

        for _epoch in 0..3 {
            let loss = assert_ok(grad.gradient(&x, &y));
            losses.push(loss);
        }

        model = grad.model().clone();

        assert!(losses.first().unwrap() > losses.last().unwrap());
    }
}
