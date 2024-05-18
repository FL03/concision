/*
    Appellation: layer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Config;
use crate::{Biased, LinearParams, ParamMode, Unbiased};
use concision::Forward;
use nd::prelude::*;
use nd::RemoveAxis;
use num::traits::{Float, FromPrimitive, One, Zero};

// #62
///
/// Layer Normalization directly estimates the normalization statistics from the summed inputs
/// to the neurons within a _hidden_ layer, eliminating the need to introduce any additional dependencies.
///
/// [LayerNorm] follows the [Layer Normalization](https://arxiv.org/abs/1607.06450) paper.
///
/// ### Resources
pub struct LayerNorm<A = f64, K = crate::Biased, D = Ix2>
where
    D: Dimension,
{
    config: Config<D>,
    params: LinearParams<A, K, D>,
}

impl<A, K, D> LayerNorm<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    pub fn from_config(config: Config<D>) -> Self
    where
        A: Default,
    {
        let params = LinearParams::<A, K, D>::new(config.dim());
        Self { config, params }
    }

    pub fn default<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let config = Config::new().dim(dim.clone()).build();
        let params = LinearParams::<A, K, D>::new(dim);
        Self { config, params }
    }

    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let config = Config::new().dim(dim.clone()).build();
        let params = LinearParams::<A, K, D>::ones(dim);
        Self { config, params }
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let config = Config::new().dim(dim.clone()).build();
        let params = LinearParams::<A, K, D>::zeros(dim);
        Self { config, params }
    }

    pub const fn config(&self) -> &Config<D> {
        &self.config
    }

    pub fn is_biased(&self) -> bool {
        self.params().is_biased()
    }
    /// Returns an immutable reference to the layer's parameters.
    pub const fn params(&self) -> &LinearParams<A, K, D> {
        &self.params
    }
    /// Returns a mutable reference to the layer's parameters.
    pub fn params_mut(&mut self) -> &mut LinearParams<A, K, D> {
        &mut self.params
    }

    pub fn dim(&self) -> D::Pattern {
        self.config().dim()
    }

    pub fn eps(&self) -> f64 {
        self.config().eps()
    }

    pub fn ndim(&self) -> usize {
        self.config().ndim()
    }

    pub fn raw_dim(&self) -> D {
        self.config().raw_dim()
    }

    pub fn shape(&self) -> &[usize] {
        self.config().shape()
    }
}

impl<A, D> Default for LayerNorm<A, Biased, D>
where
    A: Default,
    D: RemoveAxis,
{
    fn default() -> Self {
        Self {
            config: Config::default(),
            params: Default::default(),
        }
    }
}

impl<A, D> Default for LayerNorm<A, Unbiased, D>
where
    A: Default,
    D: RemoveAxis,
{
    fn default() -> Self {
        Self {
            config: Config::default(),
            params: Default::default(),
        }
    }
}

impl<A, D> Forward<Array<A, D>> for LayerNorm<A, Biased, D>
where
    A: Float + FromPrimitive,
    D: RemoveAxis,
{
    type Output = Array<A, D>;

    fn forward(&self, x: &Array<A, D>) -> Self::Output {
        let norm = if let Some(axis) = self.config().axis() {
            super::layer_norm_axis(x, *axis, self.eps())
        } else {
            super::layer_norm(x, self.eps())
        };
        norm * self.params().weights() + self.params().bias()
    }
}

impl<A, D> Forward<Array<A, D>> for LayerNorm<A, Unbiased, D>
where
    A: Float + FromPrimitive,
    D: RemoveAxis,
{
    type Output = Array<A, D>;

    fn forward(&self, x: &Array<A, D>) -> Self::Output {
        let norm = if let Some(axis) = self.config().axis() {
            super::layer_norm_axis(x, *axis, self.eps())
        } else {
            super::layer_norm(x, self.eps())
        };
        norm * self.params().weights()
    }
}
