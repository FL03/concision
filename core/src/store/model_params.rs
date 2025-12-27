/*
    Appellation: store <module>
    Contrib: @FL03
*/

use concision_params::ParamsBase;
use ndarray::{Dimension, RawData};

use crate::RawHidden;

pub struct DeepNeuralNetworkStore<X, Y, Z> {
    pub input: X,
    pub hidden: Y,
    pub output: Z,
}

/// The [`ModelParamsBase`] object is a generic container for storing the parameters of a
/// neural network, regardless of the layout (e.g. shallow or deep). This is made possible
/// through the introduction of a generic hidden layer type, `H`, that allows us to define
/// aliases and additional traits for contraining the hidden layer type. Additionally, the
/// structure enables the introduction of common accessors and initialization routines.
///
/// With that in mind, we don't reccomend using the implementation directly, rather, leverage
/// a type alias that best suites your use case (e.g. owned parameters, arc parameters, etc.).
pub struct ModelParamsBase<S, D, H, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
    H: RawHidden<S, D>,
{
    /// the input layer of the model
    pub(crate) input: ParamsBase<S, D, A>,
    /// a sequential stack of params for the model's hidden layers
    pub(crate) hidden: H,
    /// the output layer of the model
    pub(crate) output: ParamsBase<S, D, A>,
}
