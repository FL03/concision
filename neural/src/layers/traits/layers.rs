/*
    appellation: layers <module>
    authors: @FL03
*/
use super::{Activator, ActivatorGradient};

use cnc::params::ParamsBase;
use cnc::{Backward, Forward};
use ndarray::{Data, Dimension, RawData};

/// A generic trait defining the composition of a _layer_ within a neural network.
pub trait Layer<S, D>
where
    D: Dimension,
    S: RawData<Elem = Self::Elem>,
{
    /// the type of element used within the layer; typically a floating-point variant like
    /// [`f32`] or [`f64`].
    type Elem;
    /// The type of activator used by the layer; the type must implement [`ActivatorGradient`]
    type Rho: Activator<Self::Elem>;

    fn rho(&self) -> &Self::Rho;
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &ParamsBase<S, D>;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut ParamsBase<S, D>;
}
/// The [`LayerExt`] trait extends the base [`Layer`] trait with additional methods that
/// are commonly used in neural network layers. It provides methods for setting parameters,
/// performing backward propagation of errors, and completing a forward pass through the layer.
pub trait LayerExt<S, D>: Layer<S, D>
where
    D: Dimension,
    S: RawData<Elem = Self::Elem>,
{
    /// update the layer parameters
    fn set_params(&mut self, params: ParamsBase<S, D>) {
        *self.params_mut() = params;
    }
    /// backward propagate error through the layer
    fn backward<X, Y, Z, Dt>(&mut self, input: X, error: Y, gamma: Self::Elem) -> Option<Z>
    where
        S: Data,
        Self: ActivatorGradient<X, Input = Y, Output = Z, Delta = Dt>,
        Self::Elem: Clone,
        ParamsBase<S, D>: Backward<X, Dt, Elem = Self::Elem, Output = Z>,
    {
        // compute the delta using the activation function
        let delta = self.activate_gradient(error);
        // apply the backward function of the inherited layer
        self.params_mut().backward(&input, &delta, gamma)
    }
    /// complete a forward pass through the layer
    fn forward<X, Y>(&self, input: &X) -> Option<Y>
    where
        ParamsBase<S, D>: Forward<X, Output = Y>,
        Self: Activator<Y, Output = Y>,
    {
        self.params().forward_then(input, |y| self.activate(y))
    }
}
