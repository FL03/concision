/*
    Appellation: neural_network <module>
    Created At: 2025.12.10:16:20:19
    Contrib: @FL03
*/
use concision_params::RawParams;
use concision_traits::{RawStore, RawStoreMut, Store};
use ndarray::{Dimension, RawData};

pub trait NetworkParams<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}
/// The [`NetworkConfig`] trait defines an interface for compatible configurations within the
/// framework, providing a layout and a key-value store to manage hyperparameters.
pub trait NetworkConfig<K, V> {
    /// the type of key-value store used to handle the hyperparameters of the network
    type Store: RawStore<K, V>;

    /// returns a reference to the key-value store
    fn store(&self) -> &Self::Store;
    /// returns a mutable reference to the key-value store
    fn store_mut(&mut self) -> &mut Self::Store;
    /// get a reference to a value in the store by key
    fn get<'a>(&'a self, key: &K) -> Option<&'a V>
    where
        Self::Store: 'a,
    {
        self.store().get(key)
    }
    /// returns a mutable reference to a value in the store by key
    fn get_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>
    where
        Self::Store: 'a + RawStoreMut<K, V>,
    {
        self.store_mut().get_mut(key)
    }
    /// returns the entry associated with the given key
    fn hyperparam<'a>(&'a mut self, key: K) -> <Self::Store as Store<K, V>>::Entry<'a>
    where
        Self::Store: 'a + Store<K, V>,
    {
        self.store_mut().entry(key)
    }
}

/// The [`NeuralNetwork`] trait is used to define the network itself as well as each of its
/// constituent parts.
pub trait NeuralNetwork<A>
where
    Self::Params<A>: RawParams<Elem = A>,
{
    /// The configuration of the neural network defines its architecture and hyperparameters.
    type Config: NetworkConfig<String, A>;
    /// The parameters of the neural network define its weights and biases.
    type Params<_A>;

    /// returns a reference to the network configuration;
    fn config(&self) -> &Self::Config;
    /// returns a reference to the network parameters
    fn params(&self) -> &Self::Params<A>;
    /// returns a mutable reference to the network parameters
    fn params_mut(&mut self) -> &mut Self::Params<A>;
}

/// A trait defining common constants for neural networks.
pub trait NetworkConsts {
    const NAME: &'static str;
    const VERSION: &'static str;
}
