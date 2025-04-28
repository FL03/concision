/*
    Appellation: model <module>
    Contrib: @FL03
*/
//! This module provides the scaffolding for creating models and layers in a neural network.

#[doc(inline)]
pub use self::{config::StandardModelConfig, model_params::ModelParams, trainer::Trainer};

pub mod config;
pub mod model_params;
pub mod trainer;

pub(crate) mod prelude {
    pub use super::Model;
    pub use super::config::*;
    pub use super::model_params::*;
    pub use super::trainer::*;
}

use crate::{ModelFeatures, NetworkConfig, Predict, Train};
use concision_data::Dataset;

/// The base interface for all models; each model provides access to a configuration object
/// defined as the associated type [`Config`](Model::Config). The configuration object is used 
/// to provide hyperparameters and other control related parameters. In addition, the model's 
/// layout is defined by the [`features`](Model::features) method which aptly returns a copy of
/// its [ModelFeatures] object.
pub trait Model<T = f32> {
    /// The configuration type for the model
    type Config: NetworkConfig<T>;
    /// returns an immutable reference to the models configuration; this is typically used to
    /// access the models hyperparameters (i.e. learning rate, momentum, etc.) and other
    /// related control parameters.
    fn config(&self) -> &Self::Config;
    /// returns a mutable reference to the models configuration; useful for setting hyperparams
    fn config_mut(&mut self) -> &mut Self::Config;
    /// returns a copy of the models features (or layout); this is used to define the structure
    /// of the model and its consituents.
    fn features(&self) -> ModelFeatures;
    /// returns an immutable reference to the model parameters
    fn params(&self) -> &ModelParams<T>;
    /// returns a mutable reference to the model's parameters
    fn params_mut(&mut self) -> &mut ModelParams<T>;
    /// propagates the input through the model; each layer is applied in sequence meaning that
    /// the output of each previous layer is the input to the next layer. This pattern
    /// repeats until the output layer returns the final result.
    ///
    /// By default, the trait simply passes each output from one layer to the next, however,
    /// custom models will likely override this method to inject activation methods and other
    /// related logic
    fn predict<U, V>(&self, inputs: &U) -> crate::NeuralResult<V>
    where
        Self: Predict<U, Output = V>,
    {
        Predict::predict(self, inputs)
    }
    /// a convience method that trains the model using the provided dataset; this method
    /// requires that the model implements the [`Train`] trait and that the dataset
    fn train<U, V, W>(&mut self, dataset: &Dataset<U, V>) -> crate::NeuralResult<W>
    where
        Self: Train<U, V, Output = W>,
    {
        Train::train(self, dataset.records(), dataset.targets())
    }
    /// returns a model trainer prepared to train the model; this is a convenience method
    /// that creates a new trainer instance and returns it. Trainers are lazily evaluated
    /// meaning that the training process won't begin until the user calls the `begin` method.
    fn trainer<U, V>(&mut self, dataset: Dataset<U, V>) -> Trainer<'_, Self, T, Dataset<U, V>>
    where
        Self: Sized,
        T: Default,
    {
        Trainer::new(self, dataset)
    }

    #[doc(hidden)]
    #[deprecated(since = "0.1.17", note = "use predict instead")]
    fn forward<U, V>(&self, inputs: &U) -> cnc::Result<V>
    where
        Self: cnc::Forward<U, Output = V>,
    {
        <Self as cnc::Forward<U>>::forward(self, inputs)
    }
}
