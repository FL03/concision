/*
    appellation: impl_layer_repr <module>
    authors: @FL03
*/
use super::LayerBase;

use crate::layers::{Linear, ReLU, Sigmoid, Tanh};



impl<T> LayerBase<Linear, T>{
    /// initialize a new [`LayerBase`] using a [`Linear`] activation function and the given
    /// parameters.
    pub const fn linear(params: T) -> Self {
        Self {
            rho: Linear,
            params,
        }
    }
}

impl<T> LayerBase<Sigmoid, T>{
    /// initialize a new [`LayerBase`] using a [`Sigmoid`] activation function and the given
    /// parameters.
    pub const fn sigmoid(params: T) -> Self {
        Self {
            rho: Sigmoid,
            params,
        }
    }
}

impl<T> LayerBase<Tanh, T>{
    /// initialize a new [`LayerBase`] using a [`Tanh`] activation function and the given
    /// parameters.
    pub const fn tanh(params: T) -> Self {
        Self { rho: Tanh, params }
    }
}

impl<T> LayerBase<ReLU, T> {
    /// initialize a new [`LayerBase`] using a [`ReLU`] activation function and the given
    pub const fn relu(params: T) -> Self {
        Self { rho: ReLU, params }
    }
}
