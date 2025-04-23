/*
    Appellation: impl_trainer <module>
    Contrib: @FL03
*/
use crate::train::trainer::{Trainer, TrainingConfig};

impl<'a, M, T> Trainer<'a, M, T> {
    pub fn new(
        model: &'a mut M,
        learning_rate: T,
        momentum: T,
        decay: T,
        batch_size: usize,
        epochs: usize,
    ) -> Self
    where
        T: Default,
    {
        let config = TrainingConfig::new(learning_rate, momentum, decay, batch_size, epochs);
        Self {
            model,
            config,
            loss: T::default(),
        }
    }
}

impl<'a, M, T> core::ops::Deref for Trainer<'a, M, T> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.model
    }
}
impl<'a, M, T> core::ops::DerefMut for Trainer<'a, M, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.model
    }
}
