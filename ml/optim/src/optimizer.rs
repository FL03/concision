/*
    Appellation: optimizer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::params::Params;
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
