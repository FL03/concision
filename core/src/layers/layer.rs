/*
    Appellation: layer <module>
    Contrib: @FL03
*/
//! this module defines the [`LayerBase`] struct, a generic representation of a neural network
//! layer by associating some _activation function_, `F`, with some params, `T`.
//!

mod impl_layer;
mod impl_layer_deprecated;
mod impl_layer_repr;

use super::Activator;
use concision_traits::Forward;

/// The [`LayerBase`] aims to provide a generic implementation of a single layer within a
/// neural network
pub struct LayerBase<F, X> {
    /// the activation function of the layer
    pub(crate) rho: F,
    /// the parameters of the layer is an object consisting of both a weight and a bias tensor.
    pub(crate) params: X,
}

impl<F, X> LayerBase<F, X> {
    /// create a new [`LayerBase`] from the given activation function and parameters.
    pub const fn new(rho: F, params: X) -> Self {
        Self { rho, params }
    }
    /// create a new [`LayerBase`] from the given parameters assuming the logical default for
    /// the activation of type `F`.
    pub fn from_params(params: X) -> Self
    where
        F: Default,
    {
        Self {
            rho: F::default(),
            params,
        }
    }
    /// create a new [`LayerBase`] from the given activation function and shape.
    pub fn from_rho<Sh>(rho: F) -> Self
    where
        X: Default,
    {
        Self {
            rho,
            params: <X>::default(),
        }
    }
    /// returns an immutable reference to the layer's parameters
    pub const fn params(&self) -> &X {
        &self.params
    }
    /// returns a mutable reference to the layer's parameters
    pub const fn params_mut(&mut self) -> &mut X {
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
    pub fn with_params<Y>(self, params: Y) -> LayerBase<F, Y>
    where
        F: Activator<Y>,
    {
        LayerBase {
            rho: self.rho,
            params,
        }
    }
    /// consumes the current instance and returns another with the given activation function.
    /// This is useful during the creation of the model, when the activation function is not known yet.
    pub fn with_rho<G>(self, rho: G) -> LayerBase<G, X>
    where
        G: Activator<X>,
        F: Activator<X>,
    {
        LayerBase {
            rho,
            params: self.params,
        }
    }
    /// given some input, complete a single forward pass through the layer
    pub fn forward<U, V>(&self, input: &U) -> V
    where
        X: Forward<U, Output = V>,
        F: Activator<V, Output = V>,
        V: Clone,
    {
        self.params()
            .forward_then(input, |y| self.rho().activate(y))
    }
}
