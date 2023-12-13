/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;

pub trait ApplyGradient<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    fn apply_gradient(&mut self, gamma: T, gradients: &Array<T, D>);
}

pub trait Autograd<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    fn autograd(&mut self, loss: &Array<T, D>) -> Array<T, D>;
}

