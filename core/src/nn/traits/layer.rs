/*
    Appellation: layer <module>
    Created At: 2025.12.10:16:50:03
    Contrib: @FL03
*/
use concision_params::{ParamsBase, RawParams};
use concision_traits::{Activator, ActivatorGradient, Backward, Forward};
use ndarray::{Data, Dimension, RawData};

/// The [`RawLayer`] trait defines an interface for a core building block of all neural
/// networks, the layer. Layers are composed of an activation function and a set of parameters
/// that define the transformation applied to input data as it passes through the layer.
pub trait RawLayer<F, P, A = <P as RawParams>::Elem>
where
    F: Activator<A>,
    P: RawParams<Elem = A>,
{
    /// the activation function of the layer
    fn rho(&self) -> &F;
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &P;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut P;
    /// update the layer parameters
    fn set_params(&mut self, params: P) {
        *self.params_mut() = params;
    }
    /// [`replace`](core::mem::replace) the params of the layer, returning the previous value
    fn replace_params(&mut self, params: P) -> P {
        core::mem::replace(self.params_mut(), params)
    }
    /// [`swap`](core::mem::swap) the params of the layer with another
    fn swap_params(&mut self, other: &mut P) {
        core::mem::swap(self.params_mut(), other);
    }
    /// complete a forward pass through the layer
    fn forward<X, Y>(&self, input: &X) -> Y
    where
        P: Forward<X, Output = Y>,
        Self: Activator<Y, Output = Y>,
    {
        self.params().forward_then(input, |y| self.activate(y))
    }
}
/// A generic trait defining the composition of a _layer_ within a neural network.
pub trait LayerExt<F, S, D, A = <S as RawData>::Elem>: RawLayer<F, ParamsBase<S, D>, A>
where
    F: Activator<A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// backward propagate error through the layer
    fn backward<X, Y, Z, Dt>(&mut self, input: X, error: Y, gamma: A)
    where
        S: Data,
        Self: ActivatorGradient<Y, Rel = F, Delta = Dt>,
        A: Clone,
        ParamsBase<S, D>: Backward<X, Dt, Elem = A>,
    {
        let delta = self.activate_gradient(error);
        self.params_mut().backward(&input, &delta, gamma)
    }
}
