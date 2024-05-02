/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, Dimension};

pub trait Biased {
    type Bias;
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Self::Bias;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Self::Bias;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Self::Bias);
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

// macro_rules! impl_biased {
//     (@impl $name:ty($))
// }