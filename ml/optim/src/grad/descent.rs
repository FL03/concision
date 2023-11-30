/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::{Forward, Layer};
use ndarray::prelude::{Array1, Array2, NdFloat};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};

#[derive(Clone)]
pub struct GradientDescent<T = f64>
where
    T: Float,
{
    pub gamma: T,
    model: Layer<T>,
}

impl<T> GradientDescent<T>
where
    T: Float,
{
    pub fn new(gamma: T, model: Layer<T>) -> Self {
        Self { gamma, model }
    }

    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut T {
        &mut self.gamma
    }

    pub fn model(&self) -> &Layer<T> {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut Layer<T> {
        &mut self.model
    }

    pub fn set_gamma(&mut self, gamma: T) {
        self.gamma = gamma;
    }

    pub fn set_model(&mut self, model: Layer<T>) {
        self.model = model;
    }

    pub fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_model(mut self, model: Layer<T>) -> Self {
        self.model = model;
        self
    }
}

impl<T> GradientDescent<T>
where
    T: NdFloat + Signed,
{
    pub fn gradient(
        &mut self,
        data: &Array2<T>,
        targets: &Array2<T>,
        grad: impl Fn(&Array2<T>) -> Array2<T>,
    ) -> anyhow::Result<T> {
        let lr = self.gamma();
        let ns = T::from(data.shape()[0]).unwrap();
        let pred = self.model.forward(data);

        let scale = T::from(2).unwrap() * ns;

        let errors = &pred - targets;
        let dz = errors * grad(&pred);
        let dw = data.t().dot(&dz) / scale;

        self.model_mut()
            .update_with_gradient(lr, &dw.t().to_owned());

        let loss = targets.mean_sq_err(&self.model().forward(data))?;
        Ok(T::from(loss).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::prelude::{LayerShape, Objective, Sigmoid};
    use ndarray::prelude::{Array, Array2};

    fn sample_data(inputs: usize, outputs: usize, samples: usize) -> (Array2<f64>, Array2<f64>) {
        let m = samples * inputs;
        let n = samples * outputs;
        let x = Array::linspace(1., m as f64, m)
            .into_shape((samples, inputs))
            .unwrap();
        let y = Array::linspace(1., n as f64, n)
            .into_shape((samples, outputs))
            .unwrap();
        (x, y)
    }

    #[test]
    fn test_descent() {
        let (samples, inputs, outputs) = (20, 5, 3);

        let (_epochs, gamma) = (1, 0.01);
        // Generate some example data
        let (x, y) = sample_data(inputs, outputs, samples);
        let features = LayerShape::new(inputs, outputs);
        let model = Layer::from(features).init(true);

        let mut grad = GradientDescent::new(gamma, model);

        let l1 = {
            let tmp = grad.gradient(&x, &y, |xs| Sigmoid::default().gradient(xs));
            assert!(tmp.is_ok());
            tmp.unwrap()
        };

        let l2 = {
            let tmp = grad.gradient(&x, &y, |xs| Sigmoid::default().gradient(xs));
            assert!(tmp.is_ok());
            tmp.unwrap()
        };

        assert!(l1 > l2);
    }
}
