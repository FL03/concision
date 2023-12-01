/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::models::ModelParams;
use crate::neural::prelude::{Forward, LayerParams};
use ndarray::prelude::{Array2, NdFloat};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};

pub struct Grad<T = f64>
where
    T: Float,
{
    gamma: T,
    params: Vec<LayerParams<T>>,
    objective: fn(&Array2<T>) -> Array2<T>,
}

impl<T> Grad<T>
where
    T: Float,
{
    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut T {
        &mut self.gamma
    }

    pub fn objective(&self) -> fn(&Array2<T>) -> Array2<T> {
        self.objective
    }

    pub fn model(&self) -> &[LayerParams<T>] {
        &self.params
    }

    pub fn model_mut(&mut self) -> &mut [LayerParams<T>] {
        &mut self.params
    }
}

impl<T> Grad<T>
where
    T: NdFloat + Signed,
{
    pub fn step(&mut self, data: &Array2<T>, targets: &Array2<T>) -> anyhow::Result<T> {
        let grad = self.objective();
        let layers = self.model().len();
        let lr = self.gamma();
        let ns = T::from(data.shape()[0]).unwrap();

        let mut cost = T::zero();
        let params = self.params.clone();

        for (i, layer) in self.params[1..].iter_mut().enumerate() {
            let pred = params[i - 1].forward(data);
            let errors = &pred - targets;
            let dz = errors * grad(&pred);
            let dw = data.t().dot(&dz) / ns;
            layer.update_with_gradient(lr, &dw.t().to_owned());
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
