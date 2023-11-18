/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Array2, Dimension, Ix2};
use num::Float;

pub trait Initializer<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    fn init_weight(&self) -> Array<T, D>;
}

pub trait Trainable<T: Float> {
    fn train(&mut self, args: &Array2<T>, targets: &Array2<T>) -> Array2<T>;
}
