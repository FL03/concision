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
use ndarray::prelude::{Array, Array2, Dimension};

pub trait Backward<T> {
    type Params;
    type Output;

    fn backward(&mut self, args: &T, params: &Self::Params) -> Self::Output;
}

pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

pub trait Propagate<T> {
    type Optimizer;

    fn backward<D: Dimension>(
        &mut self,
        args: &Array2<T>,
        targets: &Array<T, D>,
        opt: Self::Optimizer,
    ) -> Array<T, D>;

    fn forward(&self, args: &Array2<T>) -> Array2<T>;
}

pub(crate) mod utils {}
