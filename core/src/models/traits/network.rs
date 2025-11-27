/*
    appellation: network <module>
    authors: @FL03
*/
use super::{DeepModelRepr, RawHidden, ShallowModelRepr};
use crate::config::NetworkConfig;
use ndarray::{Dimension, RawData};

pub trait NeuralNetwork<S, D>
where
    D: Dimension,
    S: RawData,
{
    type Config: NetworkConfig<S::Elem>;
    type Hidden: RawHidden<S, D>;

    fn config(&self) -> &Self::Config;
    fn config_mut(&mut self) -> &mut Self::Config;
}

pub trait ShallowNeuralNetwork<S, D>: NeuralNetwork<S, D>
where
    D: Dimension,
    S: RawData,
    Self::Hidden: ShallowModelRepr<S, D>,
{
}

pub trait DeepNeuralNetwork<S, D>: NeuralNetwork<S, D>
where
    D: Dimension,
    S: RawData,
    Self::Hidden: DeepModelRepr<S, D>,
{
}
