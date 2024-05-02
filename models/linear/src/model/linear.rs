/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Config;
use crate::params::LinearParams;
use concision::models::Module;
use concision::prelude::{Predict, PredictError};
use ndarray::{Dimension, Ix2};

pub struct Linear<T = f64, D = Ix2>
where
    D: Dimension,
{
    config: Config,
    params: LinearParams<T, D>,
}
impl<T, D> Linear<T, D>
where
    D: Dimension,
{
    pub fn new(config: Config, params: LinearParams<T, D>) -> Self {
        Self { config, params }
    }

    pub fn with_params<D2>(self, params: LinearParams<T, D2>) -> Linear<T, D2>
    where
        D2: Dimension,
    {
        Linear {
            config: self.config,
            params,
        }
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
    D: Dimension,
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
    D: Dimension,
    LinearParams<T, D>: Predict<A, Output = B>,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        self.params.predict(input)
    }
}
