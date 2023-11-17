/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::models::ModelParams;
use ndarray::prelude::{Array1, Array2, NdFloat};
use num::Float;

pub struct Grad<T = f64>
where
    T: Float,
{
    gamma: T,
    params: Vec<ModelParams<T>>,
    objective: fn(&Array2<T>) -> Array2<T>,
}

impl<T> Grad<T> where T: Float {
    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut T {
        &mut self.gamma
    }

    pub fn objective(&self) -> fn(&Array2<T>) -> Array2<T> {
        self.objective
    }

    pub fn params(&self) -> &Vec<ModelParams<T>> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Vec<ModelParams<T>> {
        &mut self.params
    }


}

impl<T> Grad<T> where T: NdFloat {

    pub fn step(&mut self, x: &Array2<T>, y: &Array2<T>) -> anyhow::Result<T> {
        let mut cost = T::zero();
        Ok(cost)
    }


}

pub struct GradStep<T = f64>
where
    T: Float,
{
    gamma: T,

    params: ModelParams<T>,
}

impl<T> GradStep<T>
where
    T: Float,
{
    pub fn new(gamma: T, params: ModelParams<T>) -> Self {
        Self { gamma, params }
    }

    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut T {
        &mut self.gamma
    }

    pub fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }

    pub fn set_gamma(&mut self, gamma: T) {
        self.gamma = gamma;
    }

    pub fn set_params(&mut self, params: ModelParams<T>) {
        self.params = params;
    }

    pub fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_params(mut self, params: ModelParams<T>) -> Self {
        self.params = params;
        self
    }
}

impl<T> GradStep<T> where T: Float {}

#[cfg(test)]
mod tests {

    #[test]
    fn test_gradient() {
        let (samples, inputs) = (20, 5);
        let _shape = (samples, inputs);
    }
}
