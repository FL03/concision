/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Params;
use ndarray::prelude::{Array1, Array2, NdFloat};
use num::Float;

pub struct Grad<T = f64>
where
    T: Float,
{
    gamma: T,
    params: Vec<Box<dyn Params>>,
    objective: fn(&T) -> T,
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

    pub fn objective(&self) -> fn(&T) -> T {
        self.objective
    }

    pub fn params(&self) -> &Vec<Box<dyn Params>> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Vec<Box<dyn Params>> {
        &mut self.params
    }
}

impl<T> Grad<T>
where
    T: NdFloat,
{
    pub fn step(&mut self, x: &Array2<T>, y: &Array1<T>) -> anyhow::Result<T> {
        let cost = T::zero();
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
