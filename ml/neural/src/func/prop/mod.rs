/*
    Appellation: prop <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Propagation
//!
//! This module describes the propagation of data through a neural network.
pub use self::{modes::*, propagation::*, utils::*};

pub(crate) mod modes;
pub(crate) mod propagation;

// pub mod forward;
use ndarray::prelude::{Array, Ix2};

pub type ForwardDyn<T = f64, D = Ix2> = Box<dyn Forward<Array<T, D>, Output = Array<T, D>>>;

pub trait Backward<T>: Forward<T> {
    type Optim;

    fn backward(&mut self, args: &T, grad: &T);
}

pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

pub(crate) mod utils {}
