/*
    appellation: impl_layer_repr <module>
    authors: @FL03
*/
use super::LayerBase;

use crate::layers::{Linear, ReLU, Sigmoid, Tanh};
use concision_params::ParamsBase;
use ndarray::{Dimension, RawData};

impl<S, D, A> LayerBase<Linear, S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// initialize a new [`LayerBase`] using a [`Linear`] activation function and the given
    /// parameters.
    pub const fn linear(params: ParamsBase<S, D, A>) -> Self {
        Self {
            rho: Linear,
            params,
        }
    }
}

impl<S, D, A> LayerBase<Sigmoid, S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// initialize a new [`LayerBase`] using a [`Sigmoid`] activation function and the given
    /// parameters.
    pub const fn sigmoid(params: ParamsBase<S, D, A>) -> Self {
        Self {
            rho: Sigmoid,
            params,
        }
    }
}

impl<S, D, A> LayerBase<Tanh, S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// initialize a new [`LayerBase`] using a [`Tanh`] activation function and the given
    /// parameters.
    pub const fn tanh(params: ParamsBase<S, D, A>) -> Self {
        Self { rho: Tanh, params }
    }
}

impl<S, D, A> LayerBase<ReLU, S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// initialize a new [`LayerBase`] using a [`ReLU`] activation function and the given
    pub const fn relu(params: ParamsBase<S, D, A>) -> Self {
        Self { rho: ReLU, params }
    }
}
