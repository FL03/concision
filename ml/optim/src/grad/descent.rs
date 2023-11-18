/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::{Forward, Linear};
use crate::prelude::Norm;
use ndarray::prelude::{Array1, Array2};
use ndarray_stats::DeviationExt;

#[derive(Clone)]
pub struct GradientDescent {
    pub gamma: f64,
    model: Linear,
}

impl GradientDescent {
    pub fn new(gamma: f64, model: Linear) -> Self {
        Self { gamma, model }
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut f64 {
        &mut self.gamma
    }

    pub fn model(&self) -> &Linear {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut Linear {
        &mut self.model
    }

    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }

    pub fn set_model(&mut self, model: Linear) {
        self.model = model;
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_model(mut self, model: Linear) -> Self {
        self.model = model;
        self
    }

    pub fn gradient(
        &mut self,
        data: &Array2<f64>,
        targets: &Array1<f64>,
        grad: impl Fn(&Array1<f64>) -> Array1<f64>,
    ) -> anyhow::Result<f64> {
        let lr = self.gamma();
        let (samples, _inputs) = data.dim();
        let pred = self.model.forward(data);

        let errors = &pred - targets;
        let dz = errors * grad(&pred);
        let dw = data.t().dot(&dz) / (2.0 * samples as f64);

        self.model_mut().update_with_gradient(lr, &dw);

        let loss = targets.mean_sq_err(&self.model().forward(data))?;
        Ok(loss)
    }

    pub fn step(&mut self, data: &Array2<f64>, targets: &Array1<f64>) -> anyhow::Result<f64> {
        // let pred = self.model.forward(data);
        let gradient = |p: &Array1<f64>| {
            let error = targets - &data.dot(&(p / p.l2()));
            let scale = -1.0 / (2.0 * data.len() as f64);
            let grad = scale * error.dot(data);

            &grad / grad.l2()
        };
        self.model.apply_gradient(self.gamma, &gradient);

        let loss = targets.mean_sq_err(&self.model.forward(data))?;
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::{Array, Array1, Array2};

    fn sample_data(samples: usize, inputs: usize) -> (Array2<f64>, Array1<f64>) {
        let n = samples * inputs;
        let x = Array::linspace(1., n as f64, n)
            .into_shape((samples, inputs))
            .unwrap();
        let y = Array::linspace(1., samples as f64, samples)
            .into_shape(samples)
            .unwrap();
        (x, y)
    }

    #[test]
    fn test_descent() {
        let (samples, inputs) = (20, 5);

        let (_epochs, gamma) = (1, 0.01);
        // Generate some example data
        let (x, y) = sample_data(samples, inputs);

        let model = Linear::new(inputs).init_weight();
        let mut grad = GradientDescent::new(gamma, model);

        let _s = grad.step(&x, &y);
    }
}
