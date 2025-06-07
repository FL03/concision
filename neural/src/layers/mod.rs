/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//! This module implments various layers for a neural network
#[doc(inline)]
pub use self::layer::LayerBase;

pub(crate) mod layer;

#[cfg(feature = "attention")]
pub mod attention;

pub(crate) mod prelude {
    #[cfg(feature = "attention")]
    pub use super::attention::prelude::*;
    pub use super::layer::*;
}

use cnc::params::ParamsBase;
use cnc::{Backward, Forward, Tensor};

use ndarray::{Data, Dimension, RawData};

/// The [`Activate`] trait defines a method for applying an activation function to an input tensor.
pub trait Activate<T> {
    type Output;

    /// Applies the activation function to the input tensor.
    fn activate(&self, input: T) -> Self::Output;
}
/// The [`ActivateGradient`] trait extends the [`Activate`] trait to include a method for 
/// computing the gradient of the activation function.
pub trait ActivateGradient<T>: Activate<T> {
    type Input;
    type Delta;

    /// compute the gradient of some input
    fn activate_gradient(&self, input: Self::Input) -> Self::Delta;
}

/// A layer within a neural-network containing a set of parameters and an activation function.
/// Here, this manifests as a wrapper around the parameters of the layer with a generic
/// activation function and corresponding traits to denote desired behaviors.
///
pub trait Layer<S, D>
where
    D: Dimension,
    S: RawData<Elem = Self::Scalar>,
{
    type Scalar;

    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &ParamsBase<S, D>;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut ParamsBase<S, D>;
    /// update the layer parameters
    fn set_params(&mut self, params: ParamsBase<S, D>) {
        *self.params_mut() = params;
    }
    /// backward propagate error through the layer
    fn backward<X, Y, Z, Delta>(
        &mut self,
        input: X,
        error: Y,
        gamma: Self::Scalar,
    ) -> cnc::Result<Z>
    where
        S: Data,
        Self: ActivateGradient<X, Input = Y, Delta = Delta>,
        Self::Scalar: Clone,
        ParamsBase<S, D>: Backward<X, Delta, Elem = Self::Scalar, Output = Z>,
    {
        // compute the delta using the activation function
        let delta = self.activate_gradient(error);
        // apply the backward function of the inherited layer
        self.params_mut().backward(&input, &delta, gamma)
    }
    /// complete a forward pass through the layer
    fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        Y: Tensor<S, D, Scalar = Self::Scalar>,
        ParamsBase<S, D>: Forward<X, Output = Y>,
        Self: Activate<Y, Output = Y>,
    {
        self.params().forward_then(input, |y| self.activate(y))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Linear;

impl<U> Activate<U> for Linear {
    type Output = U;

    fn activate(&self, x: U) -> Self::Output {
        x
    }
}

impl<U> ActivateGradient<U> for Linear
where
    U: cnc::LinearActivation,
{
    type Input = U;
    type Delta = U::Output;

    fn activate_gradient(&self, _inputs: U) -> Self::Delta {
        _inputs.linear_derivative()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sigmoid;

impl<U> Activate<U> for Sigmoid
where
    U: cnc::Sigmoid,
{
    type Output = U::Output;

    fn activate(&self, x: U) -> Self::Output {
        cnc::Sigmoid::sigmoid(x)
    }
}

impl<U> ActivateGradient<U> for Sigmoid
where
    U: cnc::Sigmoid,
{
    type Input = U;
    type Delta = U::Output;

    fn activate_gradient(&self, x: U) -> Self::Delta {
        cnc::Sigmoid::sigmoid_derivative(x)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tanh;

impl<U> Activate<U> for Tanh
where
    U: cnc::Tanh,
{
    type Output = U::Output;

    fn activate(&self, x: U) -> Self::Output {
        x.tanh()
    }
}
impl<U> ActivateGradient<U> for Tanh
where
    U: cnc::Tanh,
{
    type Input = U;
    type Delta = U::Output;

    fn activate_gradient(&self, inputs: U) -> Self::Delta {
        inputs.tanh_derivative()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReLU;

impl<U> Activate<U> for ReLU
where
    U: cnc::ReLU,
{
    type Output = U::Output;

    fn activate(&self, x: U) -> Self::Output {
        x.relu()
    }
}

impl<U> ActivateGradient<U> for ReLU
where
    U: cnc::ReLU,
{
    type Input = U;
    type Delta = U::Output;

    fn activate_gradient(&self, inputs: U) -> Self::Delta {
        inputs.relu_derivative()
    }
}
