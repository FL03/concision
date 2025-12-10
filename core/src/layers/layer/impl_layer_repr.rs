/*
    appellation: impl_layer_repr <module>
    authors: @FL03
*/
use super::Layer;

use crate::activate::{Activator, HyperbolicTangent, Linear, ReLU, Sigmoid};
use concision_params::ParamsBase;
use ndarray::{DataOwned, Dimension, RawData, RemoveAxis, ShapeBuilder};

impl<F, S, D, A> Layer<F, ParamsBase<S, D, A>>
where
    F: Activator<A, Output = A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new [`LayerBase`] from the given activation function and shape.
    pub fn from_rho_with_shape<Sh>(rho: F, shape: Sh) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            rho,
            params: ParamsBase::default(shape),
        }
    }
}

impl<T> Layer<Linear, T> {
    /// initialize a new [`LayerBase`] using a [`Linear`] activation function and the given
    /// parameters.
    pub const fn linear(params: T) -> Self {
        Self {
            rho: Linear,
            params,
        }
    }
}

impl<T> Layer<Sigmoid, T> {
    /// initialize a new [`LayerBase`] using a [`Sigmoid`] activation function and the given
    /// parameters.
    pub const fn sigmoid(params: T) -> Self {
        Self {
            rho: Sigmoid,
            params,
        }
    }
}

impl<T> Layer<HyperbolicTangent, T> {
    /// initialize a new [`LayerBase`] using a [`Tanh`] activation function and the given
    /// parameters.
    pub const fn tanh(params: T) -> Self {
        Self {
            rho: HyperbolicTangent,
            params,
        }
    }
}

impl<T> Layer<ReLU, T> {
    /// initialize a new [`LayerBase`] using a [`ReLU`] activation function and the given
    pub const fn relu(params: T) -> Self {
        Self { rho: ReLU, params }
    }
}
