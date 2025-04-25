/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//! This module implments various layers for a neural network
#[doc(inline)]
pub use self::layer::LayerBase;

pub(crate) mod layer;

#[cfg(feature = "attention")]
pub mod attention;

pub(crate) mod prelude {
    #[cfg(feature = "attention")]
    pub use super::attention::prelude::*;
    pub use super::layer::*;
}

use crate::Activate;

use ndarray::{Dimension, Ix2, RawData};
pub trait Layer<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    type Rho<U, V>: Activate<U, Output = V>;
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &cnc::ParamsBase<S, D>;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut cnc::ParamsBase<S, D>;
    /// returns an immutable reference to the activation function of the layer
    fn rho<A, B>(&self) -> &Self::Rho<A, B>;
    ///
    fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        S::Elem: Clone,
        S: ndarray::Data,
        cnc::ParamsBase<S, D>: cnc::Forward<X, Output = Y>,
    {
        self.params().forward(input).map(|y| self.rho().activate(y))
    }
}
