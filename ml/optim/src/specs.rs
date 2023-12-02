/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Forward;
use ndarray::prelude::{Array, Array2, Dimension, Ix2};
use num::Float;

pub trait ApplyGradient<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    fn apply_gradient(&mut self, gamma: T, gradients: &Array<T, D>);
}

pub trait Optimize<T = f64> {
    type Model: Forward<Array2<T>, Output = Array2<T>>;

    fn name(&self) -> &str;

    // fn optimize(&mut self, model: &mut Self::Model, args: &Array2<T>, targets: &Array2<T>) -> T {
    //     let gradients = model.backward(args, targets);
    //     let loss = model.loss(args, targets);
    //     self.update(model, &gradients);
    //     loss
    // }
}
