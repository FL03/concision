/*
    Appellation: neural_network <module>
    Created At: 2025.11.28:15:01:28
    Contrib: @FL03
*/
use super::{Deep, NetworkDepth, Shallow};
use crate::config::NetworkConfig;
use ndarray::{Dimension, RawData};

/// The [`NeuralNetwork`] trait defines a generic interface for neural network models.
pub trait NeuralNetwork<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Config: NetworkConfig<A>;
    type Depth: NetworkDepth;

    /// returns a reference to the network configuration;
    fn config(&self) -> &Self::Config;
}

pub trait DeepNeuralNetwork<S, D, A = <S as RawData>::Elem>:
    NeuralNetwork<S, D, A, Depth = Deep>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    private!();
}

pub trait ShallowNeuralNetwork<S, D, A = <S as RawData>::Elem>:
    NeuralNetwork<S, D, A, Depth = Shallow>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    private!();
}

impl<S, D, A, N> DeepNeuralNetwork<S, D, A> for N
where
    D: Dimension,
    S: RawData<Elem = A>,
    N: NeuralNetwork<S, D, A, Depth = Deep>,
{
    seal!();
}

impl<S, D, A, N> ShallowNeuralNetwork<S, D, A> for N
where
    D: Dimension,
    S: RawData<Elem = A>,
    N: NeuralNetwork<S, D, A, Depth = Shallow>,
{
    seal!();
}
