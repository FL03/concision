/*
    Appellation: config <module>
    Contrib: @FL03
*/
use crate::{model::ModelConfig, train::trainer::TrainingConfig};

impl<T> TrainingConfig<T> {
    pub fn new(learning_rate: T, momentum: T, decay: T) -> Self {
        Self {
            learning_rate,
            momentum,
            decay,
        }
    }

    pub fn from_model_config(config: ModelConfig<T>) -> Self
    where
        T: Copy + Default,
    {
        Self {
            learning_rate: config.learning_rate().copied().unwrap_or_default(),
            momentum: config.momentum().copied().unwrap_or_default(),
            decay: config.decay().copied().unwrap_or_default(),
        }
    }

    gsw! {
        learning_rate: &T,
        momentum: &T,
        decay: &T,

    }
}

impl<T> Default for TrainingConfig<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            learning_rate: T::default(),
            momentum: T::default(),
            decay: T::default(),
        }
    }
}
