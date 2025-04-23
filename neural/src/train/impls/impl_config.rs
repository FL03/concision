/*
    Appellation: config <module>
    Contrib: @FL03
*/
use crate::train::trainer::TrainingConfig;

impl<T> TrainingConfig<T> {
    pub fn new(learning_rate: T, momentum: T, decay: T, batch_size: usize, epochs: usize) -> Self {
        Self {
            learning_rate,
            momentum,
            decay,
            batch_size,
            epochs,
        }
    }
    gsw! {
        batch_size: usize,
        epochs: usize,
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
            batch_size: 0,
            epochs: 0,
        }
    }
}
