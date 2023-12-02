/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::func::activate::Sigmoid;
use crate::neural::models::ModelParams;
use crate::neural::prelude::{Forward, Gradient, LayerParams};
use ndarray::prelude::{Array2, NdFloat};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};

pub struct Grad<T = f64, O = Sigmoid>
where
    O: Gradient<T>,
    T: Float,
{
    gamma: T,
    params: Vec<LayerParams<T>>,
    objective: O,
}

impl<T, O> Grad<T, O>
where
    O: Gradient<T>,
    T: Float,
{
    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut T {
        &mut self.gamma
    }

    pub fn objective(&self) -> &O {
        &self.objective
    }

    pub fn model(&self) -> &[LayerParams<T>] {
        &self.params
    }

    pub fn model_mut(&mut self) -> &mut [LayerParams<T>] {
        &mut self.params
    }
}

impl<T, O> Grad<T, O>
where
    O: Gradient<T>,
    T: NdFloat + Signed,
{
    pub fn step(&mut self, data: &Array2<T>, targets: &Array2<T>) -> anyhow::Result<T> {
        let ns = T::from(data.shape()[0]).unwrap();

        let mut cost = T::zero();
        let params = self.params.clone();

        for (i, layer) in self.params[1..].iter_mut().enumerate() {
            // compute the prediction of the model
            let pred = params[i - 1].forward(data);
            // compute the error of the prediction
            let errors = &pred - targets;
            // compute the gradient of the objective function w.r.t. the error's
            let dz = errors * self.objective.gradient(&pred);
            // compute the gradient of the objective function w.r.t. the model's weights
            let dw = data.t().dot(&dz) / ns;
            layer.update_with_gradient(self.gamma, &dw.t().to_owned());
            let loss = targets.mean_sq_err(&pred)?;
            cost += T::from(loss).unwrap();
        }

        Ok(cost)
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
