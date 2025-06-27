/*
    Appellation: store <module>
    Contrib: @FL03
*/
mod impl_model_params;
mod impl_params_deep;
mod impl_params_shallow;

#[cfg(feature = "rand")]
mod impl_model_params_rand;

use cnc::params::ParamsBase;
use ndarray::{Dimension, Ix2, RawData};

use crate::RawHidden;

/// a type alias for an owned representation of the [`DeepParamsBase`] generic of type `A` and
/// the dimension `D`.
pub type DeepModelParams<A, D = Ix2> = DeepParamsBase<ndarray::OwnedRepr<A>, D>;

pub type DeepParamsBase<S, D> = ModelParamsBase<S, D, Vec<ParamsBase<S, D>>>;

/// a type alias for an owned representation of the [`DeepParamsBase`] generic of type `A` and
/// the dimension `D`.
pub type ShallowModelParams<A, D = Ix2> = ShallowParamsBase<ndarray::OwnedRepr<A>, D>;

pub type ShallowParamsBase<S, D> = ModelParamsBase<S, D, ParamsBase<S, D>>;

/// The [`DeepParamsBase`]
///
/// This object is an abstraction over the parameters of a deep neural network model. This is
/// done to isolate the necessary parameters from the specific logic within a model allowing us
/// to easily create additional stores for tracking velocities, gradients, and other metrics
/// we may need.
///
/// Additionally, this provides us with a way to introduce common creation routines for
/// initializing neural networks.
pub struct ModelParamsBase<S, D, H>
where
    D: Dimension,
    S: RawData,
    H: RawHidden<S, D>,
{
    /// the input layer of the model
    pub(crate) input: ParamsBase<S, D>,
    /// a sequential stack of params for the model's hidden layers
    pub(crate) hidden: H,
    /// the output layer of the model
    pub(crate) output: ParamsBase<S, D>,
}
