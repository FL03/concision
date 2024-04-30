/*
    Appellation: optimizer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::params::Params;
use ndarray::prelude::Array2;
use num::Float;

pub trait Context {
    type Config;
    type Params;
}

pub trait Optimizer<T = f64>
where
    T: Float,
{
    type Config;
    type Dataset;
    type Params;

    fn config(&self) -> &Self::Config;

    fn name(&self) -> &str;

    fn load(&mut self, dataset: &Self::Dataset);

    fn step(&mut self, params: Self::Params) -> T;
}
