/*
    appellation: impl_layer_repr <module>
    authors: @FL03
*/
use crate::layers::layer::LayerBase;

use crate::layers::{Linear, ReLU, Sigmoid, Tanh};
use cnc::ParamsBase;
use ndarray::{Dimension, RawData};

impl<S, D> LayerBase<Linear, S, D>
where
    D: Dimension,
    S: RawData<Elem = f32>,
{
    /// initialize a new [`LayerBase`] using a [`Linear`] activation function and the given
    /// parameters.
    pub const fn linear(params: ParamsBase<S, D>) -> Self {
        Self {
            rho: Linear,
            params,
        }
    }
}

impl<S, D> LayerBase<Sigmoid, S, D>
where
    D: Dimension,
    S: RawData<Elem = f32>,
{
    /// initialize a new [`LayerBase`] using a [`Sigmoid`] activation function and the given
    /// parameters.
    pub const fn sigmoid(params: ParamsBase<S, D>) -> Self {
        Self {
            rho: Sigmoid,
            params,
        }
    }
}

impl<S, D> LayerBase<Tanh, S, D>
where
    D: Dimension,
    S: RawData<Elem = f32>,
{
    /// initialize a new [`LayerBase`] using a [`Tanh`] activation function and the given
    /// parameters.
    pub const fn tanh(params: ParamsBase<S, D>) -> Self {
        Self { rho: Tanh, params }
    }
}

impl<S, D> LayerBase<ReLU, S, D>
where
    D: Dimension,
    S: RawData<Elem = f32>,
{
    /// initialize a new [`LayerBase`] using a [`ReLU`] activation function and the given
    pub const fn relu(params: ParamsBase<S, D>) -> Self {
        Self { rho: ReLU, params }
    }
}
