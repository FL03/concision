/*
    Appellation: layer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{LinearParams, ParamMode};
use nd::prelude::*;
use nd::RemoveAxis;

// #62
/// [LayerNorm] adhears to the [Layer Normalization](https://arxiv.org/abs/1607.06450) algorithm.
///
/// ### Resources
pub struct LayerNorm<A = f64, K = crate::Biased, D = Ix2>
where
    D: Dimension,
{
    config: LayerNormConfig<D>,
    params: LinearParams<A, K, D>,
}

pub struct LayerNormConfig<D = Ix2> {
    pub dim: D,
    pub eps: f64,
}

impl<D> LayerNormConfig<D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            dim: D::default(),
            eps: 1e-5,
        }
    }

    pub fn create(dim: D, eps: f64) -> Self
    where
        D: Default,
    {
        Self { dim, eps }
    }

    pub fn with_dim(dim: D) -> Self {
        Self { dim, eps: 1e-5 }
    }
}

impl<D> Default for LayerNormConfig<D>
where
    D: Default,
{
    fn default() -> Self {
        Self {
            dim: D::default(),
            eps: 1e-5,
        }
    }
}

impl<A, K, D> LayerNorm<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    pub fn from_shape<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let config = LayerNormConfig::with_dim(dim.clone());
        let params = LinearParams::<A, K, D>::default(dim);
        Self { config, params }
    }

    pub fn config(&self) -> &LayerNormConfig<D> {
        &self.config
    }

    concision::getters!(params => LinearParams<A, K, D>);
    
}
