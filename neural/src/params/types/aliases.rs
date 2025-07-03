/*
    appellation: aliases <module>
    authors: @FL03
*/
use crate::params::ModelParamsBase;
use cnc::params::ParamsBase;
use ndarray::{Ix2, OwnedRepr};

/// A type alias for an owned representation of the [`ModelParamsBase`] generic of type `A`
/// and the dimension `D`.
pub type ModelParams<A, D, H> = ModelParamsBase<OwnedRepr<A>, D, H>;
/// a type alias for an owned representation of the [`DeepParamsBase`] generic of type `A` and
/// the dimension `D`.
pub type DeepModelParams<A, D = Ix2> = DeepParamsBase<OwnedRepr<A>, D>;
/// a type alias for a _deep_ representation of the [`ModelParamsBase`] using a vector of
/// parameters as the hidden layers.
pub type DeepParamsBase<S, D> = ModelParamsBase<S, D, Vec<ParamsBase<S, D>>>;

/// a type alias for an owned representation of the [`DeepParamsBase`] generic of type `A` and
/// the dimension `D`.
pub type ShallowModelParams<A, D = Ix2> = ShallowParamsBase<OwnedRepr<A>, D>;
/// a type alias for a _shallow_ representation of the [`ModelParamsBase`] using a single
/// [`ParamsBase`] instance as the hidden layer.
pub type ShallowParamsBase<S, D> = ModelParamsBase<S, D, ParamsBase<S, D>>;
