/*
    Appellation: layer <module>
    Created At: 2025.12.10:16:50:03
    Contrib: @FL03
*/
use crate::activate::{Activator, ActivatorGradient};
use concision_params::RawParams;
use concision_traits::{Backward, Forward};

/// The [`RawLayer`] trait establishes a common interface for all _layers_ within a given
/// model. Implementors will need to define the type of parameters they utilize, as well as
/// provide methods to access both the activation function and the parameters of the layer.
pub trait RawLayer<F, A>
where
    F: Activator<A>,
    Self::Params<A>: RawParams<Elem = A>,
{
    type Params<_T>;
    /// the activation function of the layer
    fn rho(&self) -> &F;
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &Self::Params<A>;
    /// complete a forward pass through the layer
    fn forward<X, Y, Z>(&self, input: &X) -> Z
    where
        F: Activator<Y, Output = Z>,
        Self::Params<A>: Forward<X, Output = Y>,
    {
        let y = self.params().forward(input);
        self.rho().activate(y)
    }
}
/// The [`RawLayerMut`] trait extends the [`RawLayer`] trait by providing mutable access to the
/// layer's parameters and additional methods for training the layer, such as backward
/// propagation and parameter updates.
pub trait RawLayerMut<F, A>: RawLayer<F, A>
where
    F: Activator<A>,
    Self::Params<A>: RawParams<Elem = A>,
{
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut Self::Params<A>;
    /// backward propagate error through the layer
    fn backward<X, Y, Z, Dt>(&mut self, input: X, error: Y, gamma: A)
    where
        A: Clone,
        F: ActivatorGradient<Y, Rel = F, Delta = Dt>,
        Self::Params<A>: Backward<X, Dt, Elem = A>,
    {
        let delta = self.rho().activate_gradient(error);
        self.params_mut().backward(&input, &delta, gamma)
    }
    /// update the layer parameters
    fn set_params(&mut self, params: Self::Params<A>) {
        *self.params_mut() = params;
    }
    /// [`replace`](core::mem::replace) the params of the layer, returning the previous value
    fn replace_params(&mut self, params: Self::Params<A>) -> Self::Params<A> {
        core::mem::replace(self.params_mut(), params)
    }
    /// [`swap`](core::mem::swap) the params of the layer with another
    fn swap_params(&mut self, other: &mut Self::Params<A>) {
        core::mem::swap(self.params_mut(), other);
    }
}
