/*
    Appellation: config <module>
    Contrib: @FL03
*/

use crate::Hyperparameters::*;

pub type ModelConfigMap<T> = std::collections::HashMap<String, T>;

pub struct ModelConfigBuilder<T = f32> {
    pub learning_rate: Option<T>,
}

pub struct ModelConfig<T> {
    pub(crate) batch_size: usize,
    pub(crate) epochs: usize,
    pub(crate) hyperparameters: ModelConfigMap<T>,
}

impl<T> ModelConfig<T> {
    pub fn new() -> Self {
        Self {
            batch_size: 0,
            epochs: 0,
            hyperparameters: ModelConfigMap::new(),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn with_batch_size(self, batch_size: usize) -> Self {
        Self { batch_size, ..self }
    }

    pub fn with_epochs(self, epochs: usize) -> Self {
        Self { epochs, ..self }
    }

    pub fn insert_hyperparameter(&mut self, key: impl ToString, value: T) -> Option<T> {
        self.hyperparameters.insert(key.to_string(), value)
    }

    pub fn set_decay(&mut self, decay: T) -> Option<T> {
        self.insert_hyperparameter(Decay, decay)
    }
    pub fn set_learning_rate(&mut self, learning_rate: T) -> Option<T> {
        self.insert_hyperparameter(LearningRate, learning_rate)
    }

    pub fn set_momentum(&mut self, momentum: T) -> Option<T> {
        self.insert_hyperparameter(Momentum, momentum)
    }

    pub fn set_weight_decay(&mut self, decay: T) -> Option<T> {
        self.insert_hyperparameter("weight_decay", decay)
    }

    pub fn get(&self, key: impl ToString) -> Option<&T> {
        self.hyperparameters.get(&key.to_string())
    }

    pub fn learning_rate(&self) -> Option<&T> {
        self.get(LearningRate)
    }

    pub fn momentum(&self) -> Option<&T> {
        self.get(Momentum)
    }

    pub fn decay(&self) -> Option<&T> {
        self.get(Decay)
    }
}
