/*
    Appellation: neural_network <module>
    Created At: 2025.11.28:15:01:28
    Contrib: @FL03
*/
use ndarray::{Dimension, RawData};

/// The [`NetworkDepth`] trait is used to define the depth/kind of a neural network model.
pub trait NetworkDepth {
    private!();
}

unit_types! {
    #[NetworkDepth]
    pub enum {
        Deep,
        Shallow,
    }
}

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
    /// The depth/kind of the neural network.
    type Depth: NetworkDepth;
    /// The parameters of the neural network define its weights and biases.
    type Params<_S, _D>: NeuralNetworkParams<_S, _D, A>
    where
        _S: RawData<Elem = A>,
        _D: Dimension;

    /// returns a reference to the network configuration;
    fn config(&self) -> &Self::Config;
    fn params(&self) -> &Self::Params<S, D>;
}

/// A compact trait the procedural macro can implement to expose network metadata.
pub trait NetworkConst {
    const NAME: &'static str;
    const VERSION: &'static str;
}
