/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Config, Layout};
use crate::{Biased, LinearParams, ParamMode, Unbiased};
use concision::prelude::{Predict, Result};
use nd::prelude::*;
use nd::RemoveAxis;

/// An implementation of a linear model.
///
/// In an effort to streamline the api, the [Linear] model relies upon a [ParamMode] type ([Biased] or [Unbiased](crate::params::mode::Unbiased))
/// which enables the model to automatically determine whether or not to include a bias term. Doing so allows the model to inherit several methods
/// familar to the underlying [ndarray](https://docs.rs/ndarray) crate.
pub struct Linear<A = f64, K = Biased, D = Ix2>
where
    D: Dimension,
{
    pub(crate) config: Config<K, D>,
    pub(crate) params: LinearParams<A, K, D>,
}

impl<A, K, D> Linear<A, K, D>
where
    D: RemoveAxis,
{
    impl_model_builder!(default where A: Default);
    impl_model_builder!(ones where A: Clone + num::One);
    impl_model_builder!(zeros where A: Clone + num::Zero);

    pub fn from_config(config: Config<K, D>) -> Self
    where
        A: Clone + Default,
        K: ParamMode,
    {
        let params = LinearParams::default(config.dim());
        Self { config, params }
    }

    pub fn from_layout(layout: Layout<D>) -> Self
    where
        A: Clone + Default,
        K: ParamMode,
    {
        let config = Config::<K, D>::new().with_layout(layout);
        let params = LinearParams::default(config.dim());
        Self { config, params }
    }

    /// Applies an activcation function onto the prediction of the model.
    pub fn activate<X, Y, F>(&self, args: &X, func: F) -> Result<Y>
    where
        F: for<'a> Fn(&'a Y) -> Y,
        Self: Predict<X, Output = Y>,
    {
        Ok(func(&self.predict(args)?))
    }

    pub const fn config(&self) -> &Config<K, D> {
        &self.config
    }

    pub fn weights(&self) -> &Array<A, D> {
        self.params.weights()
    }

    pub fn weights_mut(&mut self) -> &mut Array<A, D> {
        self.params.weights_mut()
    }

    pub const fn params(&self) -> &LinearParams<A, K, D> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut LinearParams<A, K, D> {
        &mut self.params
    }

    pub fn into_biased(self) -> Linear<A, Biased, D>
    where
        A: Default,
        K: 'static,
    {
        Linear {
            config: self.config.into_biased(),
            params: self.params.into_biased(),
        }
    }

    pub fn into_unbiased(self) -> Linear<A, Unbiased, D>
    where
        A: Default,
        K: 'static,
    {
        Linear {
            config: self.config.into_unbiased(),
            params: self.params.into_unbiased(),
        }
    }

    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        self.config().is_biased()
    }

    pub fn with_params<E>(self, params: LinearParams<A, K, E>) -> Linear<A, K, E>
    where
        E: RemoveAxis,
    {
        let config = self.config.into_dimensionality(params.raw_dim()).unwrap();
        Linear { config, params }
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            config: self.config.with_name(name),
            ..self
        }
    }
}

impl<A, D> Linear<A, Biased, D>
where
    D: RemoveAxis,
{
    pub fn biased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        let config = Config::<Biased, D>::new().with_shape(shape);
        let params = LinearParams::biased(config.dim());
        Linear { config, params }
    }

    pub fn bias(&self) -> &Array<A, D::Smaller> {
        self.params().bias()
    }

    pub fn bias_mut(&mut self) -> &mut Array<A, D::Smaller> {
        self.params_mut().bias_mut()
    }
}

impl<A, D> Linear<A, Unbiased, D>
where
    D: RemoveAxis,
{
    pub fn unbiased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        let config = Config::<Unbiased, D>::new().with_shape(shape);
        let params = LinearParams::unbiased(config.dim());
        Linear { config, params }
    }
}
