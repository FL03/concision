/*
    Appellation: layer <module>
    Contrib: @FL03
*/
mod impl_layer;
mod impl_layer_deprecated;
mod impl_layer_repr;

use super::Activator;
use concision_traits::Forward;

/// The [`Layer`] implementation works to provide a generic interface for layers within a
/// neural network. It associates an activation function of type `F` with parameters of
/// type `P`.
pub struct Layer<F, P> {
    /// the activation function of the layer
    pub(crate) rho: F,
    /// the parameters of the layer is an object consisting of both a weight and a bias tensor.
    pub(crate) params: P,
}

impl<F, P> Layer<F, P> {
    /// create a new [`Layer`] from the given activation function and parameters.
    pub const fn new(rho: F, params: P) -> Self {
        Self { rho, params }
    }
    /// create a new [`Layer`] from the given parameters assuming the logical default for
    /// the activation of type `F`.
    pub fn from_params(params: P) -> Self
    where
        F: Default,
    {
        Self {
            rho: F::default(),
            params,
        }
    }
    /// create a new [`Layer`] from the given activation function and shape.
    pub fn from_rho<Sh>(rho: F) -> Self
    where
        P: Default,
    {
        Self {
            rho,
            params: <P>::default(),
        }
    }
    /// returns an immutable reference to the layer's parameters
    pub const fn params(&self) -> &P {
        &self.params
    }
    /// returns a mutable reference to the layer's parameters
    pub const fn params_mut(&mut self) -> &mut P {
        &mut self.params
    }
    /// returns an immutable reference to the activation function of the layer
    pub const fn rho(&self) -> &F {
        &self.rho
    }
    /// returns a mutable reference to the activation function of the layer
    pub const fn rho_mut(&mut self) -> &mut F {
        &mut self.rho
    }
    /// consumes the current instance and returns another with the given parameters.
    pub fn with_params<Y>(self, params: Y) -> Layer<F, Y>
    where
        F: Activator<Y>,
    {
        Layer {
            rho: self.rho,
            params,
        }
    }
    /// consumes the current instance and returns another with the given activation function.
    /// This is useful during the creation of the model, when the activation function is not known yet.
    pub fn with_rho<G>(self, rho: G) -> Layer<G, P>
    where
        G: Activator<P>,
        F: Activator<P>,
    {
        Layer {
            rho,
            params: self.params,
        }
    }
    /// given some input, complete a single forward pass through the layer
    pub fn forward<U, V>(&self, input: &U) -> V
    where
        P: Forward<U, Output = V>,
        F: Activator<V, Output = V>,
        V: Clone,
    {
        self.params()
            .forward_then(input, |y| self.rho().activate(y))
    }
}
