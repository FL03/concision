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
/// a type alias for a _deep_ representation of the [`ModelParamsBase`] using a vector of
/// parameters as the hidden layers.
pub type DeepParamsBase<S, D> = ModelParamsBase<S, D, Vec<ParamsBase<S, D>>>;

/// a type alias for an owned representation of the [`DeepParamsBase`] generic of type `A` and
/// the dimension `D`.
pub type ShallowModelParams<A, D = Ix2> = ShallowParamsBase<ndarray::OwnedRepr<A>, D>;
/// a type alias for a _shallow_ representation of the [`ModelParamsBase`] using a single
/// [`ParamsBase`] instance as the hidden layer.
pub type ShallowParamsBase<S, D> = ModelParamsBase<S, D, ParamsBase<S, D>>;

/// The [`ModelParamsBase`] object is a generic ocntainer for storing the parameters of a
/// neural network, regardless of the layout (e.g. shallow or deep). This is made possible
/// through the introduction of a generic hidden layer type, `H`, that allows us to define
/// aliases and additional traits for contraining the hidden layer type. That being said, we
/// don't reccoment using this type directly, but rather use the provided type aliases such as
/// [`DeepModelParams`] or [`ShallowModelParams`] or their owned variants. These provide a much
/// more straighforward interface for typing the parameters of a neural network. We aren't too
/// worried about the transumtation between the two since users desiring this ability should
/// simply stick with a _deep_ representation, initializing only a single layer within the
/// respective container.
///
/// This type also enables us to define a set of common initialization routines and introduce
/// other standards for dealing with parameters in a neural network.
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
