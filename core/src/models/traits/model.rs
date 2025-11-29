/*
    appellation: models <module>
    authors: @FL03
*/
use crate::config::ModelConfiguration;
use crate::{DeepModelParams, ModelLayout, RawModelLayout};
use concision_params::Params;
use concision_traits::Predict;

/// The [`Model`] trait defines the core interface for all models; implementors will need to
/// provide the type of configuration used by the model, the type of layout used by the model,
/// and the type of parameters used by the model. The crate provides standard, or default,
/// definitions of both the configuration and layout types, however, for
pub trait Model<T = f32> {
    /// The type of configuration used for the model
    type Config: ModelConfiguration<T>;
    /// The type of [`ModelLayout`] used by the model for this implementation.
    type Layout: ModelLayout;
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
    fn layout(&self) -> &Self::Layout;
    /// returns an immutable reference to the model parameters
    fn params(&self) -> &DeepModelParams<T>;
    /// returns a mutable reference to the model's parameters
    fn params_mut(&mut self) -> &mut DeepModelParams<T>;
    /// propagates the input through the model; each layer is applied in sequence meaning that
    /// the output of each previous layer is the input to the next layer. This pattern
    /// repeats until the output layer returns the final result.
    ///
    /// By default, the trait simply passes each output from one layer to the next, however,
    /// custom models will likely override this method to inject activation methods and other
    /// related logic
    fn predict<U, V>(&self, inputs: &U) -> V
    where
        Self: Predict<U, Output = V>,
    {
        Predict::predict(self, inputs)
    }
}

pub trait ModelExt<T>: Model<T> {
    /// [`replace`](core::mem::replace) the current configuration and returns the old one;
    fn replace_config(&mut self, config: Self::Config) -> Self::Config {
        core::mem::replace(self.config_mut(), config)
    }
    /// [`replace`](core::mem::replace) the current model parameters and returns the old one
    fn replace_params(&mut self, params: DeepModelParams<T>) -> DeepModelParams<T> {
        core::mem::replace(self.params_mut(), params)
    }
    /// overrides the current configuration and returns a mutable reference to the model
    fn set_config(&mut self, config: Self::Config) -> &mut Self {
        *self.config_mut() = config;
        self
    }
    /// overrides the current model parameters and returns a mutable reference to the model
    fn set_params(&mut self, params: DeepModelParams<T>) -> &mut Self {
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
