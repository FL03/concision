/*
    Appellation: layer <module>
    Contrib: @FL03
*/
mod impl_layer;
mod impl_layer_deprecated;
mod impl_layer_repr;

use crate::activate::{Activator, ActivatorGradient};
use concision_params::{ParamsBase, RawParam};
use concision_traits::{Backward, Forward};
use ndarray::{Data, Dimension, RawData};

/// The [`Layer`] implementation works to provide a generic interface for layers within a
/// neural network. It associates an activation function of type `F` with parameters of
/// type `P`.
pub struct Layer<F, P> {
    /// the activation function of the layer
    pub(crate) rho: F,
    /// the parameters of the layer is an object consisting of both a weight and a bias tensor.
    pub(crate) params: P,
}

pub trait RawLayer<F, X, A = <X as RawParam>::Elem>
where
    F: Activator<X>,
    X: RawParam<Elem = A>,
{
    /// the activation function of the layer
    fn rho(&self) -> &F;
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &X;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut X;
}
/// A generic trait defining the composition of a _layer_ within a neural network.
pub trait NdLayer<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// The type of activator used by the layer; the type must implement [`ActivatorGradient`]
    type Rho: Activator<A>;

    fn rho(&self) -> &Self::Rho;
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &ParamsBase<S, D>;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut ParamsBase<S, D>;

    /// update the layer parameters
    fn set_params(&mut self, params: ParamsBase<S, D>) {
        *self.params_mut() = params;
    }
    /// backward propagate error through the layer
    fn backward<X, Y, Z, Dt>(&mut self, input: X, error: Y, gamma: A)
    where
        S: Data,
        Self: ActivatorGradient<Y, Output = Z, Delta = Dt>,
        A: Clone,
        ParamsBase<S, D>: Backward<X, Dt, Elem = A>,
    {
        let delta = self.activate_gradient(error);
        self.params_mut().backward(&input, &delta, gamma)
    }
    /// complete a forward pass through the layer
    fn forward<X, Y>(&self, input: &X) -> Y
    where
        ParamsBase<S, D>: Forward<X, Output = Y>,
        Self: Activator<Y, Output = Y>,
    {
        self.params().forward_then(input, |y| self.activate(y))
    }
}
