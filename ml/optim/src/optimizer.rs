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
    type Config;

    fn config(&self) -> &Self::Config;

    fn name(&self) -> &str;

    fn step(&mut self, data: &Array2<T>, targets: &Array2<T>) -> T;

    fn step_with(
        &mut self,
        data: &Array2<T>,
        targets: &Array2<T>,
        params: &mut Box<dyn Params<T>>,
    ) -> T;
}

pub trait OptimizerExt<T = f64>: Optimizer<T>
where
    T: Float,
{
}
