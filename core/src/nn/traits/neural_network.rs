/*
    Appellation: neural_network <module>
    Created At: 2025.12.10:16:20:19
    Contrib: @FL03
*/

use ndarray::{Dimension, RawData};

pub trait NeuralNetworkParams<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}

/// The [`NeuralNetwork`] trait is used to define the network itself as well as each of its
/// constituent parts.
pub trait NeuralNetwork<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// The context of the neural network defines any additional information required for its operation.
    type Ctx;
    /// The configuration of the neural network defines its architecture and hyperparameters.
    type Config;
    /// The parameters of the neural network define its weights and biases.
    type Params<_S, _D>: NeuralNetworkParams<_S, _D, A>
    where
        _S: RawData<Elem = A>,
        _D: Dimension;

    /// returns a reference to the network configuration;
    fn config(&self) -> &Self::Config;

    fn params(&self) -> &Self::Params<S, D>;

    fn params_mut(&mut self) -> &mut Self::Params<S, D>;
}

/// A trait defining common constants for neural networks.
pub trait NetworkConsts {
    const NAME: &'static str;
    const VERSION: &'static str;
}
