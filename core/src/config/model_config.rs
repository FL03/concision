/*
    Appellation: config <module>
    Contrib: @FL03
*/

mod impl_hash_config;

use super::HyperParams;
use super::{ExtendedModelConfig, ModelConfiguration, RawConfig};

#[cfg(all(feature = "alloc", not(feature = "std")))]
pub(crate) type ModelConfigMap<T> = alloc::collections::BTreeMap<HyperParams, T>;
#[cfg(feature = "std")]
pub(crate) type ModelConfigMap<T> = std::collections::HashMap<HyperParams, T>;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde_derive::Deserialize, serde::Serialize))]
pub struct StandardModelConfig<T> {
    pub(crate) batch_size: usize,
    pub(crate) epochs: usize,
    pub(crate) hyperparameters: ModelConfigMap<T>,
}

impl<T> Default for StandardModelConfig<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StandardModelConfig<T> {
    pub fn new() -> Self {
        Self {
            batch_size: 0,
            epochs: 0,
            hyperparameters: ModelConfigMap::new(),
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
    pub const fn hyperparameters(&self) -> &ModelConfigMap<T> {
        &self.hyperparameters
    }
    /// returns a mutable reference to the hyperparameters map
    pub const fn hyperparameters_mut(&mut self) -> &mut ModelConfigMap<T> {
        &mut self.hyperparameters
    }
    /// inserts a hyperparameter into the map, returning the previous value if it exists
    pub fn add_parameter<P: Into<HyperParams>>(&mut self, key: P, value: T) -> Option<T> {
        self.hyperparameters_mut().insert(key.into(), value)
    }
    /// gets a reference to a hyperparameter by key, returning None if it does not exist
    pub fn get_parameter<Q>(&self, key: &Q) -> Option<&T>
    where
        Q: ?Sized + Eq + core::hash::Hash,
        HyperParams: core::borrow::Borrow<Q>,
    {
        self.hyperparameters().get(key)
    }
    /// returns an entry for the hyperparameter, allowing for insertion or modification
    pub fn parameter<Q>(&mut self, key: Q) -> std::collections::hash_map::Entry<'_, HyperParams, T>
    where
        Q: AsRef<str>,
    {
        self.hyperparameters_mut().entry(key.as_ref().into())
    }
    /// removes a hyperparameter from the map, returning the value if it exists
    pub fn remove_hyperparameter<Q>(&mut self, key: &Q) -> Option<T>
    where
        Q: ?Sized + core::hash::Hash + Eq,
        HyperParams: core::borrow::Borrow<Q>,
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

use HyperParams::*;

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
        self.get_parameter(&LearningRate)
    }
    /// returns a reference to the momentum hyperparameter, if it exists
    pub fn momentum(&self) -> Option<&T> {
        self.get_parameter(&Momentum)
    }
    /// returns a reference to the decay hyperparameter, if it exists
    pub fn decay(&self) -> Option<&T> {
        self.get_parameter(&Decay)
    }
    /// returns a reference to the weight decay hyperparameter, if it exists
    pub fn weight_decay(&self) -> Option<&T> {
        self.get_parameter(&WeightDecay)
    }
}

unsafe impl<T> Send for StandardModelConfig<T> where T: Send {}

unsafe impl<T> Sync for StandardModelConfig<T> where T: Sync {}

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

    fn keys(&self) -> Vec<HyperParams> {
        self.hyperparameters().keys().cloned().collect()
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
