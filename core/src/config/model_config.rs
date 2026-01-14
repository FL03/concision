/*
    Appellation: config <module>
    Contrib: @FL03
*/
use super::HyperParam;
use super::{ExtendedModelConfig, ModelConfiguration, RawConfig};
use alloc::string::{String, ToString};
use hashbrown::DefaultHashBuilder;
use hashbrown::hash_map::{self, HashMap};

/// The [`StandardModelConfig`] struct is a standard implementation of the
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename = "snake_case")
)]
pub struct StandardModelConfig<T> {
    pub batch_size: usize,
    pub epochs: usize,
    pub hyperspace: HashMap<String, T>,
}

impl<T> StandardModelConfig<T> {
    pub fn new() -> Self {
        Self {
            batch_size: 0,
            epochs: 0,
            hyperspace: HashMap::new(),
        }
    }
    /// returns a copy of the batch size
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
    /// returns a mutable reference to the batch size
    pub const fn batch_size_mut(&mut self) -> &mut usize {
        &mut self.batch_size
    }
    /// returns a copy of the epochs
    pub const fn epochs(&self) -> usize {
        self.epochs
    }
    /// returns a mutable reference to the epochs
    pub const fn epochs_mut(&mut self) -> &mut usize {
        &mut self.epochs
    }
    /// returns a reference to the hyperparameters map
    pub const fn hyperparameters(&self) -> &HashMap<String, T> {
        &self.hyperspace
    }
    /// returns a mutable reference to the hyperparameters map
    pub const fn hyperparameters_mut(&mut self) -> &mut HashMap<String, T> {
        &mut self.hyperspace
    }
    /// inserts a hyperparameter into the map, returning the previous value if it exists
    pub fn add_parameter<K: ToString>(&mut self, key: K, value: T) -> Option<T> {
        self.hyperparameters_mut().insert(key.to_string(), value)
    }
    /// gets a reference to a hyperparameter by key, returning None if it does not exist
    pub fn get<Q>(&self, key: &Q) -> Option<&T>
    where
        Q: ?Sized + Eq + core::hash::Hash,
        String: core::borrow::Borrow<Q>,
    {
        self.hyperparameters().get(key)
    }
    /// returns an entry for the hyperparameter, allowing for insertion or modification
    pub fn parameter<Q: ToString>(
        &mut self,
        key: Q,
    ) -> hash_map::Entry<'_, String, T, DefaultHashBuilder> {
        self.hyperparameters_mut().entry(key.to_string())
    }
    /// removes a hyperparameter from the map, returning the value if it exists
    pub fn remove_hyperparameter<Q>(&mut self, key: &Q) -> Option<T>
    where
        Q: ?Sized + core::hash::Hash + Eq,
        String: core::borrow::Borrow<Q>,
    {
        self.hyperparameters_mut().remove(key)
    }
    /// sets the batch size, returning a mutable reference to the current instance
    pub fn set_batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = batch_size;
        self
    }
    /// sets the number of epochs, returning a mutable reference to the current instance
    pub fn set_epochs(&mut self, epochs: usize) -> &mut Self {
        self.epochs = epochs;
        self
    }
    /// consumes the current instance to create another with the given batch size
    pub fn with_batch_size(self, batch_size: usize) -> Self {
        Self { batch_size, ..self }
    }
    /// consumes the current instance to create another with the given epochs
    pub fn with_epochs(self, epochs: usize) -> Self {
        Self { epochs, ..self }
    }
}

use HyperParam::*;

impl<T> StandardModelConfig<T> {
    /// sets the decay hyperparameter, returning the previous value if it exists
    pub fn set_decay(&mut self, decay: T) -> Option<T> {
        self.add_parameter(Decay, decay)
    }
    pub fn set_learning_rate(&mut self, learning_rate: T) -> Option<T> {
        self.add_parameter(LearningRate, learning_rate)
    }
    /// sets the momentum hyperparameter, returning the previous value if it exists
    pub fn set_momentum(&mut self, momentum: T) -> Option<T> {
        self.add_parameter(Momentum, momentum)
    }
    /// sets the weight decay hyperparameter, returning the previous value if it exists
    pub fn set_weight_decay(&mut self, decay: T) -> Option<T> {
        self.add_parameter(WeightDecay, decay)
    }
    /// returns a reference to the learning rate hyperparameter, if it exists
    pub fn learning_rate(&self) -> Option<&T> {
        self.get("learning_rate")
    }
    /// returns a reference to the momentum hyperparameter, if it exists
    pub fn momentum(&self) -> Option<&T> {
        self.get("momentum")
    }
    /// returns a reference to the decay hyperparameter, if it exists
    pub fn decay(&self) -> Option<&T> {
        self.get("decay")
    }
    /// returns a reference to the weight decay hyperparameter, if it exists
    pub fn weight_decay(&self) -> Option<&T> {
        self.get("weight_decay")
    }
}

impl<T> Default for StandardModelConfig<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T> Send for StandardModelConfig<T> where T: Send {}

unsafe impl<T> Sync for StandardModelConfig<T> where T: Sync {}

impl<T> crate::nn::NetworkConfig<String, T> for StandardModelConfig<T> {
    type Store = HashMap<String, T, DefaultHashBuilder>;

    fn store(&self) -> &Self::Store {
        &self.hyperspace
    }

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.hyperspace
    }
}

impl<T> RawConfig for StandardModelConfig<T> {
    type Ctx = T;
}

impl<T> ModelConfiguration<T> for StandardModelConfig<T> {
    fn get<K>(&self, key: K) -> Option<&T>
    where
        K: AsRef<str>,
    {
        self.hyperparameters().get(key.as_ref())
    }

    fn get_mut<K>(&mut self, key: K) -> Option<&mut T>
    where
        K: AsRef<str>,
    {
        self.hyperparameters_mut().get_mut(key.as_ref())
    }

    fn set<K>(&mut self, key: K, value: T) -> Option<T>
    where
        K: AsRef<str>,
    {
        self.hyperparameters_mut()
            .insert(key.as_ref().into(), value)
    }

    fn remove<K>(&mut self, key: K) -> Option<T>
    where
        K: AsRef<str>,
    {
        self.hyperparameters_mut().remove(key.as_ref())
    }

    fn contains<K>(&self, key: K) -> bool
    where
        K: AsRef<str>,
    {
        self.hyperparameters().contains_key(key.as_ref())
    }

    fn keys(&self) -> Vec<&str> {
        self.hyperparameters().keys().map(|k| k.as_str()).collect()
    }
}

impl<T> ExtendedModelConfig<T> for StandardModelConfig<T> {
    fn epochs(&self) -> usize {
        self.epochs
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}
