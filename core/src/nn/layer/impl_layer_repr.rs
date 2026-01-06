/*
    appellation: impl_layer_repr <module>
    authors: @FL03
*/
use super::LayerBase;

use concision_params::{ParamsBase, RawParams};
use concision_traits::{Activator, HyperbolicTangent, Linear, ReLU, Sigmoid};
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, RemoveAxis, ShapeBuilder};

impl<F, S, D, A> LayerBase<F, ArrayBase<S, D, A>>
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

    pub fn dim(&self) -> D::Pattern {
        self.params().dim()
    }

    pub fn raw_dim(&self) -> D {
        self.params().raw_dim()
    }

    pub fn shape(&self) -> &[usize] {
        self.params().shape()
    }
}

impl<F, S, D, E, A> LayerBase<F, ParamsBase<S, D, A>>
where
    F: Activator<A, Output = A>,
    D: Dimension<Smaller = E>,
    E: Dimension<Larger = D>,
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

    pub const fn bias(&self) -> &ArrayBase<S, E, A> {
        self.params().bias()
    }

    pub const fn bias_mut(&mut self) -> &mut ArrayBase<S, E, A> {
        self.params_mut().bias_mut()
    }

    pub const fn weights(&self) -> &ArrayBase<S, D, A> {
        self.params().weights()
    }

    pub const fn weights_mut(&mut self) -> &mut ArrayBase<S, D, A> {
        self.params_mut().weights_mut()
    }

    pub fn dim(&self) -> D::Pattern {
        self.params().dim()
    }

    pub fn raw_dim(&self) -> D {
        self.params().raw_dim()
    }

    pub fn shape(&self) -> &[usize] {
        self.params().shape()
    }
}

impl<F, P, A> LayerBase<F, P>
where
    F: Fn(A) -> A,
    P: RawParams<Elem = A>,
{
}

impl<A, P> LayerBase<Linear, P>
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

impl<A, P> LayerBase<Sigmoid, P>
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

impl<A, P> LayerBase<HyperbolicTangent, P>
where
    P: RawParams<Elem = A>,
{
    /// initialize a new layer using a [`TanhActivator`] activation function and the given
    /// parameters.
    pub const fn tanh(params: P) -> Self {
        Self {
            rho: HyperbolicTangent,
            params,
        }
    }
}

impl<A, P> LayerBase<ReLU, P>
where
    P: RawParams<Elem = A>,
{
    /// initialize a layer using the [`Sigmoid`] activation function and the given params.
    pub const fn relu(params: P) -> Self {
        Self { rho: ReLU, params }
    }
}
