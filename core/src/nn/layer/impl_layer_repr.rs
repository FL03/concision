/*
    appellation: impl_layer_repr <module>
    authors: @FL03
*/
use super::Layer;

use crate::activate::{Activator, HyperbolicTangent, Linear, ReLU, Sigmoid};
use concision_params::{ParamsBase, RawParams};
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, RemoveAxis, ShapeBuilder};

impl<F, S, D, A> Layer<F, ArrayBase<S, D, A>>
where
    F: Activator<A, Output = A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new instance from the given activation function and shape.
    pub fn from_rho_with_shape<Sh>(rho: F, shape: Sh) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            rho,
            params: ArrayBase::default(shape),
        }
    }
}

impl<F, S, D, A> Layer<F, ParamsBase<S, D, A>>
where
    F: Activator<A, Output = A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new layer from the given activation function and shape.
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

impl<F, P, A> Layer<F, P>
where
    F: Fn(A) -> A,
    P: RawParams<Elem = A>,
{
}

impl<A, P> Layer<Linear, P>
where
    P: RawParams<Elem = A>,
{
    /// initialize a layer using the [`Linear`] activation function and the given params.
    pub const fn linear(params: P) -> Self {
        Self {
            rho: Linear,
            params,
        }
    }
}

impl<A, P> Layer<Sigmoid, P>
where
    P: RawParams<Elem = A>,
{
    /// initialize a layer using the [`Sigmoid`] activation function and the given params.
    pub const fn sigmoid(params: P) -> Self {
        Self {
            rho: Sigmoid,
            params,
        }
    }
}

impl<A, P> Layer<HyperbolicTangent, P>
where
    P: RawParams<Elem = A>,
{
    /// initialize a new layer using a [`HyperbolicTangent`] activation function and the given
    /// parameters.
    pub const fn tanh(params: P) -> Self {
        Self {
            rho: HyperbolicTangent,
            params,
        }
    }
}

impl<A, P> Layer<ReLU, P>
where
    P: RawParams<Elem = A>,
{
    /// initialize a layer using the [`Sigmoid`] activation function and the given params.
    pub const fn relu(params: P) -> Self {
        Self { rho: ReLU, params }
    }
}
