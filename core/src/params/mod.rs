/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{group::*, iter::*, kinds::*, param::*, store::*};

pub(crate) mod group;
pub(crate) mod iter;
pub(crate) mod kinds;
pub(crate) mod param;
pub(crate) mod store;

use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;
use std::collections::HashMap;

pub trait Param {
    type Dim: Dimension;

    fn kind(&self) -> &ParamKind;

    fn name(&self) -> &str;
}

pub trait Biased<T = f64>
where
    T: Float,
{
    type Dim: Dimension;
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, Self::Dim>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, Self::Dim>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, Self::Dim>);
}

pub trait Weighted<T = f64>
where
    T: Float,
{
    type Dim: Dimension;
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, Self::Dim>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, Self::Dim>;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, Self::Dim>);
}

pub trait Params<K = String, T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
    Self: IntoIterator<Item = (K, Array<T, D>)>,
{
    fn get(&self, param: &K) -> Option<&Array<T, D>>;

    fn get_mut(&mut self, param: &K) -> Option<&mut Array<T, D>>;
}

impl<K, T, D> Params<K, T, D> for HashMap<K, Array<T, D>>
where
    D: Dimension,
    K: std::cmp::Eq + std::hash::Hash,
    T: Float,
{
    fn get(&self, param: &K) -> Option<&Array<T, D>> {
        self.get(param)
    }

    fn get_mut(&mut self, param: &K) -> Option<&mut Array<T, D>> {
        self.get_mut(param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linarr;
    use ndarray::linalg::Dot;
    use ndarray::prelude::{Ix1, Ix2};

    #[test]
    fn test_parameter() {
        let a = linarr::<f64, Ix1>((3,)).unwrap();
        let p = linarr::<f64, Ix2>((3, 3)).unwrap();
        let mut param = Parameter::<f64, Ix2>::new((10, 1), ParamKind::Bias, "bias");
        param.set_params(p.clone());

        assert_eq!(param.kind(), &ParamKind::Bias);
        assert_eq!(param.name(), "bias");
        assert_eq!(param.dot(&a), p.dot(&a));
    }
}
