/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::Config;
use crate::params::LinearParams;
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


