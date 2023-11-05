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

pub trait Backward<T> {
    type Params;
    type Output;

    fn backward(&mut self, args: &T, params: &Self::Params) -> Self::Output;
}

pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

pub(crate) mod utils {}
