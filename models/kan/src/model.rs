/*
    appellation: model <module>
    authors: @FL03
*/

#[cfg(feature = "rand")]
use cnc::init::rand_distr;
use cnc::nn::{Model, ModelFeatures, ModelParams, StandardModelConfig};

use num_traits::{Float, FromPrimitive};

pub struct KanModel<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: ModelParams<T>,
}

impl<T> KanModel<T> {
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self
    where
        T: Clone + Default,
    {
        let params = ModelParams::default(features);
        KanModel {
            config,
            features,
            params,
        }
    }
    #[cfg(feature = "rand")]
    pub fn init(self) -> Self
    where
        T: Float + FromPrimitive,
        rand_distr::StandardNormal: rand_distr::Distribution<T>,
    {
        let params = ModelParams::glorot_normal(self.features);
        KanModel { params, ..self }
    }

    pub const fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }

    pub const fn features(&self) -> ModelFeatures {
        self.features
    }

    pub const fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
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

    fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }
}
