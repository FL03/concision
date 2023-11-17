/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{bias::*, param::*, shapes::*, utils::*, weight::*};

pub(crate) mod bias;
pub(crate) mod param;
pub(crate) mod shapes;
pub(crate) mod weight;

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Float;

pub trait Biased<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
    Self: Weighted<T, D>,
{
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, D::Smaller>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller>;
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
}

pub trait WeightedExt<T = f64, D = Ix2>: Weighted<T, D>
where
    Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
    D: Dimension,
    T: Float,
{
}

pub trait Params<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, D::Smaller>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller>;
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, D::Smaller>);
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, D>);
}

pub trait Parameterized<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Features: IntoDimension<Dim = D>;
    type Params: Params<T, D>;

    fn features(&self) -> &Self::Features;

    fn features_mut(&mut self) -> &mut Self::Features;

    fn params(&self) -> &Self::Params;

    fn params_mut(&mut self) -> &mut Self::Params;

    fn set_bias(&mut self, bias: Array<T, D::Smaller>) {
        self.params_mut().set_bias(bias);
    }

    fn set_weights(&mut self, weights: Array<T, D>) {
        self.params_mut().set_weights(weights);
    }
}

impl<T, D, P> Biased<T, D> for P
where
    D: Dimension,
    P: Parameterized<T, D>,
    T: Float,
    <D as Dimension>::Smaller: Dimension,
    <P as Parameterized<T, D>>::Params: 'static,
{
    fn bias(&self) -> &Array<T, D::Smaller> {
        self.params().bias()
    }

    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
        self.params_mut().bias_mut()
    }
}

impl<T, D, P> Weighted<T, D> for P
where
    P: Parameterized<T, D>,
    D: Dimension,
    T: Float,
{
    fn weights(&self) -> &Array<T, D> {
        self.params().weights()
    }

    fn weights_mut(&mut self) -> &mut Array<T, D> {
        self.params_mut().weights_mut()
    }
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
