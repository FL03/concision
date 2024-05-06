/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::Config;
use crate::params::LinearParams;
use concision::prelude::{Module, Predict, PredictError};
use ndarray::{Ix2, RemoveAxis};

/// Linear model
pub struct Linear<T = f64, D = Ix2>
where
    D: RemoveAxis,
{
    pub(crate) config: Config,
    pub(crate) params: LinearParams<T, D>,
}

impl<T, D> Linear<T, D>
where
    D: RemoveAxis,
{
    pub fn new(config: Config, params: LinearParams<T, D>) -> Self {
        Self { config, params }
    }

    pub fn with_params<D2>(self, params: LinearParams<T, D2>) -> Linear<T, D2>
    where
        D2: RemoveAxis,
    {
        Linear {
            config: self.config,
            params,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn params(&self) -> &LinearParams<T, D> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut LinearParams<T, D> {
        &mut self.params
    }

    pub fn is_biased(&self) -> bool {
        self.params().is_biased() || self.config.biased
    }
}

impl<T> Linear<T> {
    pub fn std(config: Config) -> Self
    where
        T: Clone + Default,
    {
        let params = LinearParams::new(config.biased, config.shape);
        Self { config, params }
    }
}

impl<T, D> Module for Linear<T, D>
where
    D: RemoveAxis,
{
    type Config = Config;
    type Params = LinearParams<T, D>;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Self::Params {
        &mut self.params
    }
}

impl<A, B, T, D> Predict<A> for Linear<T, D>
where
    D: RemoveAxis,
    LinearParams<T, D>: Predict<A, Output = B>,
{
    type Output = B;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip(self, input), level = "debug", name = "Linear::predict")
    )]
    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        #[cfg(feature = "tracing")]
        tracing::debug!("Predicting with linear model");
        self.params.predict(input)
    }
}
