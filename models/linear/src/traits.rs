/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, Dimension};

pub trait Biased<T = f64> {
    type Dim: Dimension;
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, Self::Dim>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, Self::Dim>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, Self::Dim>);
}

pub trait Weighted<T = f64> {
    type Dim: Dimension;
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, Self::Dim>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, Self::Dim>;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, Self::Dim>);
}
