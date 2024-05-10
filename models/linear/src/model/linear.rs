/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Config, Layout};
use crate::{Biased, LinearParams, ParamMode};
use concision::prelude::{Predict, Result};
use nd::{Array, Dimension, Ix2, RemoveAxis};

/// Linear model
pub struct Linear<K = Biased, A = f64, D = Ix2>
where
    D: Dimension,
{
    pub(crate) config: Config<D, K>,
    pub(crate) params: LinearParams<K, A, D>,
}

impl<A, D, K> Linear<K, A, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    pub fn from_config(config: Config<D, K>) -> Self
    where
        A: Clone + Default,
        K: 'static,
    {
        let params = LinearParams::default(config.dim());
        Self { config, params }
    }

    pub fn from_layout(layout: Layout<D>) -> Self
    where
        A: Clone + Default,
    {
        let config = Config::<D, K>::new().with_layout(layout);
        let params = LinearParams::default(config.dim());
        Self { config, params }
    }

    pub fn with_params<E>(self, params: LinearParams<K, A, E>) -> Linear<K, A, E>
    where
        E: RemoveAxis,
    {
        let config = self.config.into_dimensionality(params.raw_dim()).unwrap();
        Linear { config, params }
    }

    pub fn activate<X, Y, F>(&self, args: &X, func: F) -> Result<Y>
    where
        F: for<'a> Fn(&'a Y) -> Y,
        Self: Predict<X, Output = Y>,
    {
        Ok(func(&self.predict(args)?))
    }

    pub const fn config(&self) -> &Config<D, K> {
        &self.config
    }

    pub fn weights(&self) -> &Array<A, D> {
        self.params.weights()
    }

    pub fn weights_mut(&mut self) -> &mut Array<A, D> {
        self.params.weights_mut()
    }

    pub const fn params(&self) -> &LinearParams<K, A, D> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut LinearParams<K, A, D> {
        &mut self.params
    }

    pub fn into_biased(self) -> Linear<Biased, A, D>
    where
        A: Default,
    {
        Linear {
            config: self.config.into_biased(),
            params: self.params.into_biased(),
        }
    }

    pub fn is_biased(&self) -> bool {
        K::BIASED || self.config().is_biased()
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            config: self.config.with_name(name),
            ..self
        }
    }
}

impl<A, D> Linear<Biased, A, D>
where
    D: RemoveAxis,
{
    pub fn bias(&self) -> &Array<A, D::Smaller> {
        self.params.bias().unwrap()
    }

    pub fn bias_mut(&mut self) -> &mut Array<A, D::Smaller> {
        self.params.bias_mut().unwrap()
    }
}
