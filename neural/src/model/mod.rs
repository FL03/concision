/*
    Appellation: model <module>
    Contrib: @FL03
*/
//! This module provides the scaffolding for creating models and layers in a neural network.

#[doc(inline)]
pub use self::{config::ModelConfig, store::ModelParams};

pub mod config;
pub mod store;

pub(crate) mod prelude {
    pub use super::store::*;
}

use crate::ModelFeatures;

/// This trait defines the base interface for all models, providing access to the models
/// configuration, layout, and learned parameters.
pub trait Model<T = f32> {
    /// returns an immutable reference to the models configuration; this is typically used to
    /// access the models hyperparameters (i.e. learning rate, momentum, etc.) and other
    /// related control parameters.
    fn config(&self) -> &ModelConfig<T>;
    /// returns a mutable reference to the models configuration; useful for setting hyperparams
    fn config_mut(&mut self) -> &mut ModelConfig<T>;
    /// returns a copy of the models features (or layout); this is used to define the structure
    /// of the model and its consituents.
    fn features(&self) -> ModelFeatures;
    /// returns an immutable reference to the model parameters
    fn params(&self) -> &ModelParams<T>;
    /// returns a mutable reference to the model's parameters
    fn params_mut(&mut self) -> &mut ModelParams<T>;
    /// returns a model trainer prepared to train the model; this is a convenience method
    /// that creates a new trainer instance and returns it. Trainers are lazily evaluated
    /// meaning that the training process won't begin until the user calls the `begin` method.
    fn train(&mut self) -> crate::train::trainer::Trainer<'_, Self, T>
    where
        Self: Sized,
        T: Default,
    {
        crate::train::trainer::Trainer::new(self)
    }
    /// propagates the input through the model; each layer is applied in sequence meaning that
    /// the output of each previous layer is the input to the next layer. This pattern
    /// repeats until the output layer returns the final result.
    ///
    /// By default, the trait simply passes each output from one layer to the next, however,
    /// custom models will likely override this method to inject activation methods and other
    /// related logic
    fn forward<U, V>(&self, inputs: &U) -> cnc::CncResult<V>
    where
        Self: cnc::Forward<U, Output = V>,
    {
        <Self as cnc::Forward<U>>::forward(self, inputs)
    }
}
