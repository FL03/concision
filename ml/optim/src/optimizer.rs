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

    fn load(&mut self, data: &Array2<T>, targets: &Array2<T>);

    fn step(&mut self, params: impl Params) -> T;
}

pub trait OptimizerExt<T = f64>: Optimizer<T>
where
    T: Float,
{
}

pub struct Opt<T = f64> {
    epochs: usize,
    gamma: T,
}
