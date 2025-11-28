/*
    appellation: model <module>
    authors: @FL03
*/

use cnc::nn::{DeepModelParams, Model, ModelFeatures, StandardModelConfig};
#[cfg(feature = "rand")]
use cnc::rand_distr;

use num_traits::{Float, FromPrimitive};

#[derive(Clone, Debug)]
pub struct KanModel<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: DeepModelParams<T>,
}

impl<T> KanModel<T> {
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self
    where
        T: Clone + Default,
    {
        let params = DeepModelParams::default(features);
        KanModel {
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
    pub fn set_config(&mut self, config: StandardModelConfig<T>) -> &mut Self {
        self.config = config;
        self
    }
    /// set the current features and return a mutable reference to the model
    pub fn set_features(&mut self, features: ModelFeatures) -> &mut Self {
        self.features = features;
        self
    }
    /// set the current parameters and return a mutable reference to the model
    pub fn set_params(&mut self, params: DeepModelParams<T>) -> &mut Self {
        self.params = params;
        self
    }
    /// consumes the current instance to create another with the given configuration
    pub fn with_config(self, config: StandardModelConfig<T>) -> Self {
        Self { config, ..self }
    }
    /// consumes the current instance to create another with the given features
    pub fn with_features(self, features: ModelFeatures) -> Self {
        Self { features, ..self }
    }
    /// consumes the current instance to create another with the given parameters
    pub fn with_params(self, params: DeepModelParams<T>) -> Self {
        Self { params, ..self }
    }
}

impl<T> KanModel<T>
where
    T: 'static + Float + FromPrimitive,
{
    #[cfg(feature = "rand")]
    pub fn init(self) -> Self
    where
        T: 'static + Float + FromPrimitive,
        rand_distr::StandardNormal: rand_distr::Distribution<T>,
    {
        let params = DeepModelParams::glorot_normal(self.features());
        KanModel { params, ..self }
    }
}

impl<T> Model<T> for KanModel<T> {
    type Config = StandardModelConfig<T>;
    type Layout = ModelFeatures;

    fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }

    fn layout(&self) -> ModelFeatures {
        self.features
    }

    fn params(&self) -> &DeepModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut DeepModelParams<T> {
        &mut self.params
    }
}
