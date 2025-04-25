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

use crate::{Activate, ActivateGradient};
use cnc::{Backward, Forward};



pub trait Layer<T> {
    type Params;
    type Rho<U, V>: Activate<U, Output = V>;

    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &Self::Params;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut Self::Params;
    /// update the layer parameters
    fn set_params(&mut self, params: Self::Params) {
        *self.params_mut() = params;
    }
    /// returns an immutable reference to the activation function of the layer
    fn rho<A, B>(&self) -> &Self::Rho<A, B>;
    ///
    fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        Self::Params: Forward<X, Output = Y>,
    {
        self.params().forward_then(input, |y| self.rho().activate(y))
    }
    /// backward propagate 
    fn backward<X, Y, Z, Delta>(
        &mut self,
        input: &X,
        error: &Y,
        gamma: T,
    ) -> cnc::Result<Z>
    where
        Self::Params: Backward<X, Delta, HParam = T, Output = Z>,
        Self::Rho<X, Delta>: ActivateGradient<Y, Input = X, Delta = Delta>,
    {
        let delta = self.rho().activate_gradient(error);
        self.params_mut().backward(input, &delta, gamma)
    }
}
