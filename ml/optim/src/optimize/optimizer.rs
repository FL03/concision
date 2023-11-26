/*
    Appellation: optimizer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Params;
use ndarray::prelude::Array2;
use num::Float;

pub trait Optimizer<T = f64>
where
    T: Float,
{
    fn name(&self) -> &str;

    fn step(
        &mut self,
        data: &Array2<T>,
        targets: &Array2<T>,
    ) -> impl Fn(&mut Box<dyn Params<T>>) -> T;
}

pub struct OptimizerStep<T = f64>
where
    T: Float,
{
    data: Array2<T>,
    params: Vec<Box<dyn Params>>,
    targets: Array2<T>,
}

impl<T> OptimizerStep<T>
where
    T: Float,
{
    pub fn new(data: Array2<T>, targets: Array2<T>) -> Self {
        Self {
            data,
            params: Vec::new(),
            targets,
        }
    }

    pub fn zeros(inputs: usize, outputs: usize, samples: usize) -> Self {
        Self {
            data: Array2::zeros((samples, inputs)),
            params: Vec::new(),
            targets: Array2::zeros((samples, outputs)),
        }
    }

    pub fn params(&self) -> &[Box<dyn Params>] {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut [Box<dyn Params>] {
        &mut self.params
    }

    pub fn set_params(&mut self, params: Vec<Box<dyn Params>>) {
        self.params = params;
    }

    pub fn with_params(mut self, params: Vec<Box<dyn Params>>) -> Self {
        self.params = params;
        self
    }
}
