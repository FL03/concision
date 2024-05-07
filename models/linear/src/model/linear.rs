/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::Config;
use crate::params::LinearParams;
use concision::prelude::{Predict, Result};
use nd::{Array, Ix2, RemoveAxis};

/// Linear model
pub struct Linear<T = f64, D = Ix2>
where
    D: RemoveAxis,
{
    pub(crate) config: Config<D>,
    pub(crate) params: LinearParams<T, D>,
}

impl<T, D> Linear<T, D>
where
    D: RemoveAxis,
{
    pub fn new(config: Config<D>) -> Self
    where
        T: Clone + Default,
    {
        let params = LinearParams::default(config.is_biased(), config.dim());
        Self { config, params }
    }

    pub fn with_params<D2>(self, params: LinearParams<T, D2>) -> Linear<T, D2>
    where
        D2: RemoveAxis,
    {
        let config = self.config.into_dimensionality(params.raw_dim()).unwrap();
        Linear { config, params }
    }

    pub fn config(&self) -> &Config<D> {
        &self.config
    }

    pub fn bias(&self) -> Option<&Array<T, D::Smaller>> {
        self.params.bias()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Array<T, D::Smaller>> {
        self.params.bias_mut()
    }

    pub fn weights(&self) -> &Array<T, D> {
        self.params.weights()
    }

    pub fn weights_mut(&mut self) -> &mut Array<T, D> {
        self.params.weights_mut()
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

    pub fn activate<X, Y, F>(&self, args: &X, func: F) -> Result<Y>
    where
        F: for<'a> Fn(&'a Y) -> Y,
        Self: Predict<X, Output = Y>,
    {
        Ok(func(&self.predict(args)?))
    }
}
