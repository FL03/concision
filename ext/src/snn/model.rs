/*
    appellation: model <module>
    authors: @FL03
*/
use cnc::config::StandardModelConfig;
use cnc::prelude::{DeepModelParams, Model, ModelFeatures};

#[cfg(feature = "rand")]
use cnc::init::rand_distr::{Distribution, StandardNormal};
use num_traits::{Float, FromPrimitive};

#[derive(Clone, Debug)]
pub struct SpikingNeuralNetwork<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: DeepModelParams<T>,
}

impl<T> SpikingNeuralNetwork<T> {
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self
    where
        T: Clone + Default,
    {
        let params = DeepModelParams::default(features);
        SpikingNeuralNetwork {
            config,
            features,
            params,
        }
    }
    /// returns a reference to the model configuration
    pub const fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }
    /// returns a mutable reference to the model configuration
    pub const fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }
    /// returns the model features
    pub const fn features(&self) -> ModelFeatures {
        self.features
    }
    /// returns a mutable reference to the model features
    pub const fn features_mut(&mut self) -> &mut ModelFeatures {
        &mut self.features
    }
    /// returns a reference to the model parameters
    pub const fn params(&self) -> &DeepModelParams<T> {
        &self.params
    }
    /// returns a mutable reference to the model parameters
    pub const fn params_mut(&mut self) -> &mut DeepModelParams<T> {
        &mut self.params
    }
    /// set the current configuration and return a mutable reference to the model
    pub fn set_config(&mut self, config: StandardModelConfig<T>) {
        self.config = config;
    }
    /// set the current features and return a mutable reference to the model
    pub const fn set_features(&mut self, features: ModelFeatures) {
        self.features = features;
    }
    /// set the current parameters and return a mutable reference to the model
    pub fn set_params(&mut self, params: DeepModelParams<T>) {
        self.params = params;
    }
    #[inline]
    /// consumes the current instance to create another with the given configuration
    pub fn with_config(self, config: StandardModelConfig<T>) -> Self {
        Self { config, ..self }
    }
    #[inline]
    /// consumes the current instance to create another with the given features
    pub fn with_features(self, features: ModelFeatures) -> Self {
        Self { features, ..self }
    }
    #[inline]
    /// consumes the current instance to create another with the given parameters
    pub fn with_params(self, params: DeepModelParams<T>) -> Self {
        Self { params, ..self }
    }

    #[cfg(feature = "rand")]
    pub fn init(self) -> Self
    where
        T: 'static + Float + FromPrimitive,
        StandardNormal: Distribution<T>,
    {
        let params = DeepModelParams::glorot_normal(self.features());
        SpikingNeuralNetwork { params, ..self }
    }
}

impl<T> Model<T> for SpikingNeuralNetwork<T> {
    type Config = StandardModelConfig<T>;

    type Layout = ModelFeatures;

    fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }

    fn layout(&self) -> &ModelFeatures {
        &self.features
    }

    fn params(&self) -> &DeepModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut DeepModelParams<T> {
        &mut self.params
    }
}
