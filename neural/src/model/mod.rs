/*
    Appellation: model <module>
    Contrib: @FL03
*/
//! This module provides the scaffolding for creating models and layers in a neural network.

#[doc(inline)]
pub use self::{
    config::StandardModelConfig, layout::*, model_params::ModelParams, trainer::Trainer,
};

pub mod config;
pub mod layout;
pub mod model_params;
pub mod trainer;

pub(crate) mod prelude {
    pub use super::Model;
    pub use super::config::*;
    pub use super::layout::*;
    pub use super::model_params::*;
    pub use super::trainer::*;
}

use crate::{NetworkConfig, Predict, Train};
use concision_core::params::Params;
use concision_data::DatasetBase;

/// The base interface for all models; each model provides access to a configuration object
/// defined as the associated type [`Config`](Model::Config). The configuration object is used
/// to provide hyperparameters and other control related parameters. In addition, the model's
/// layout is defined by the [`features`](Model::features) method which aptly returns a copy of
/// its [ModelFeatures] object.
pub trait Model<T = f32> {
    /// The configuration type for the model
    type Config: NetworkConfig<T>;
    /// the type of layout used by the model
    type Layout;
    /// returns an immutable reference to the models configuration; this is typically used to
    /// access the models hyperparameters (i.e. learning rate, momentum, etc.) and other
    /// related control parameters.
    fn config(&self) -> &Self::Config;
    /// returns a mutable reference to the models configuration; useful for setting hyperparams
    fn config_mut(&mut self) -> &mut Self::Config;
    /// returns a copy of the model's current layout (features); a type providing the model
    /// with a particular number of features for the various layers of a deep neural network.
    ///
    /// the layout is used in everything from creation and initialization routines to
    /// validating the dimensionality of the model's inputs, outputs, training data, etc.
    fn layout(&self) -> Self::Layout;
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
    fn train<U, V, W>(&mut self, dataset: &DatasetBase<U, V>) -> crate::NeuralResult<W>
    where
        Self: Train<U, V, Output = W>,
    {
        Train::train(self, dataset.records(), dataset.targets())
    }
}

pub trait ModelExt<T>: Model<T>
where
    Self::Layout: ModelLayout,
{
    /// [`replace`](core::mem::replace) the current configuration and returns the old one;
    fn replace_config(&mut self, config: Self::Config) -> Self::Config {
        core::mem::replace(self.config_mut(), config)
    }
    /// [`replace`](core::mem::replace) the current model parameters and returns the old one
    fn replace_params(&mut self, params: ModelParams<T>) -> ModelParams<T> {
        core::mem::replace(self.params_mut(), params)
    }
    /// overrides the current configuration and returns a mutable reference to the model
    fn set_config(&mut self, config: Self::Config) -> &mut Self {
        *self.config_mut() = config;
        self
    }
    /// overrides the current model parameters and returns a mutable reference to the model
    fn set_params(&mut self, params: ModelParams<T>) -> &mut Self {
        *self.params_mut() = params;
        self
    }
    /// returns an immutable reference to the input layer;
    #[inline]
    fn input_layer(&self) -> &Params<T> {
        self.params().input()
    }
    /// returns a mutable reference to the input layer;
    #[inline]
    fn input_layer_mut(&mut self) -> &mut Params<T> {
        self.params_mut().input_mut()
    }
    /// returns an immutable reference to the hidden layer(s);
    #[inline]
    fn hidden_layers(&self) -> &Vec<Params<T>> {
        self.params().hidden()
    }
    /// returns a mutable reference to the hidden layer(s);
    #[inline]
    fn hidden_layers_mut(&mut self) -> &mut Vec<Params<T>> {
        self.params_mut().hidden_mut()
    }
    /// returns an immutable reference to the output layer;
    #[inline]
    fn output_layer(&self) -> &Params<T> {
        self.params().output()
    }
    /// returns a mutable reference to the output layer;
    #[inline]
    fn output_layer_mut(&mut self) -> &mut Params<T> {
        self.params_mut().output_mut()
    }
    #[inline]
    fn set_input_layer(&mut self, layer: Params<T>) -> &mut Self {
        self.params_mut().set_input(layer);
        self
    }
    #[inline]
    fn set_hidden_layers(&mut self, layers: Vec<Params<T>>) -> &mut Self {
        self.params_mut().set_hidden(layers);
        self
    }
    #[inline]
    fn set_output_layer(&mut self, layer: Params<T>) -> &mut Self {
        self.params_mut().set_output(layer);
        self
    }
    /// returns a 2-tuple representing the dimensions of the input layer; (input, hidden)
    fn input_dim(&self) -> (usize, usize) {
        self.layout().dim_input()
    }
    /// returns a 2-tuple representing the dimensions of the hidden layers; (hidden, hidden)
    fn hidden_dim(&self) -> (usize, usize) {
        self.layout().dim_hidden()
    }
    /// returns the total number of hidden layers in the model;
    fn hidden_layers_count(&self) -> usize {
        self.layout().layers()
    }
    /// returns a 2-tuple representing the dimensions of the output layer; (hidden, output)
    fn output_dim(&self) -> (usize, usize) {
        self.layout().dim_output()
    }
}

impl<M, T> ModelExt<T> for M
where
    M: Model<T>,
    M::Layout: ModelLayout,
{
}

/// The [`DeepNeuralNetwork`] trait is a specialization of the [`Model`] trait that
/// provides additional functionality for deep neural networks. This trait is
pub trait DeepNeuralNetwork<T = f32>: Model<T> {}

pub trait ModelTrainer<T> {
    type Model: Model<T>;
    /// returns a model trainer prepared to train the model; this is a convenience method
    /// that creates a new trainer instance and returns it. Trainers are lazily evaluated
    /// meaning that the training process won't begin until the user calls the `begin` method.
    fn trainer<'a, U, V>(
        &mut self,
        dataset: DatasetBase<U, V>,
        model: &'a mut Self::Model,
    ) -> Trainer<'a, Self::Model, T, DatasetBase<U, V>>
    where
        Self: Sized,
        T: Default,
        for<'b> &'b mut Self::Model: Model<T>,
    {
        Trainer::new(model, dataset)
    }
}
