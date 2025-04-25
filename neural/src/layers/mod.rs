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
use cnc::{Backward, Forward, ParamsBase, traits::tensor::Tensor};
use ndarray::{Data, Dimension, RawData};

/// A layer within a neural-network containing a set of parameters and an activation function.
/// Here, this manifests as a wrapper around the parameters of the layer with a generic
/// activation function and corresponding traits to denote desired behaviors.
///
pub trait Layer<S, D>
where
    D: Dimension,
    S: RawData<Elem = Self::Scalar>,
{
    type Scalar;
    type Rho<U>: Activate<U, Output = U>;

    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &ParamsBase<S, D>;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut ParamsBase<S, D>;
    /// update the layer parameters
    fn set_params(&mut self, params: ParamsBase<S, D>) {
        *self.params_mut() = params;
    }
    /// returns an immutable reference to the activation function of the layer
    fn rho<U>(&self) -> &Self::Rho<U>;
    ///
    fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        Y: Tensor<S, D, Scalar = Self::Scalar>,
        ParamsBase<S, D>: Forward<X, Output = Y>,
    {
        self.params()
            .forward_then(input, |y| self.rho().activate(y))
    }
    /// backward propagate error through the layer
    fn backward<X, Y, Z, Delta>(
        &mut self,
        input: &X,
        error: &Y,
        gamma: Self::Scalar,
    ) -> cnc::Result<Z>
    where
        ParamsBase<S, D>: Backward<X, Delta, HParam = Self::Scalar, Output = Z>,
        Self::Rho<Y>: ActivateGradient<Y, Input = X, Delta = Delta>,
        Self::Scalar: Clone,
        S: Data,
    {
        // compute the delta using the activation function
        let delta = self.rho().activate_gradient(error);
        // apply the backward function of the inherited layer
        self.params_mut().backward(input, &delta, gamma)
    }
}
