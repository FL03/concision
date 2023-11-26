/*
    Appellation: optimize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # optimize
//!
pub use self::{optimizer::*, utils::*};

pub(crate) mod optimizer;

use crate::neural::prelude::Forward;
use ndarray::prelude::{Array, Array2, Dimension, Ix2};
use num::Float;

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

pub trait Gradient<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    fn update(&mut self, gamma: T, params: &mut Array<T, D>, gradients: &Array<T, D>)
    where
        T: 'static,
    {
        params.scaled_add(-gamma, gradients);
    }
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
