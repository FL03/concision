/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::Config;
use crate::params::LinearParams;
use concision::prelude::{Predict, Result};
use nd::{Ix2, RemoveAxis};

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
    // pub fn new(biased: bool, dim: impl IntoDimension<Dim = D>) -> Self
    // where
    //     T: Clone + Default,
    // {
    //     let config = Config::from_dimension(dim);
    //     let params = LinearParams::new(config.biased, config.shape);
    //     Self { config, params }
    // }
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
        self.config().is_biased() || self.params().is_biased()
    }

    pub fn activate<Y, F>(&self, args: &nd::Array<T, D>, func: F) -> Result<Y> where F: for <'a> Fn(&'a Y) -> Y, Self: Predict<nd::Array<T, D>, Output = Y> {
        Ok(func(&self.predict(args)?))
    }
}


