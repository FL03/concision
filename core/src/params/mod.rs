/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{kinds::*, param::*, utils::*};

pub(crate) mod kinds;
pub(crate) mod param;

use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;

pub trait Parameter<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference of the param
    fn param(&self) -> &Array<T, D>;
    /// Returns a mutable reference of the param
    fn param_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the param
    fn set_param(&mut self, param: Array<T, D>);
}

pub trait Biased<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, D::Smaller>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, D::Smaller>);
}

pub trait Weighted<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, D>);
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
