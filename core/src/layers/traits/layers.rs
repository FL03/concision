/*
    appellation: layers <module>
    authors: @FL03
*/
use super::{Activator, ActivatorGradient};

use concision_params::ParamsBase;
use concision_traits::{Backward, Forward};
use ndarray::{Data, Dimension, RawData};

pub trait RawLayer<F, X> {
    type Elem;
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
