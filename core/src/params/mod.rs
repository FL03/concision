/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{iter::*, kinds::*, param::*, variable::*};

pub(crate) mod iter;
pub(crate) mod kinds;
pub(crate) mod param;
pub(crate) mod variable;

pub mod masks;
pub mod store;

use ndarray::{Array, ArrayBase, Data, Dimension};
use std::collections::HashMap;

pub trait Param {
    type Dim: Dimension;

    fn kind(&self) -> &ParamKind;

    fn name(&self) -> &str;
}

pub trait Biased<T = f64> {
    type Dim: Dimension;
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, Self::Dim>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, Self::Dim>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, Self::Dim>);
}

pub trait Weighted<T = f64> {
    type Dim: Dimension;
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, Self::Dim>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, Self::Dim>;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, Self::Dim>);
}

pub trait Params<K, V> {
    fn get(&self, param: &K) -> Option<&V>;

    fn get_mut(&mut self, param: &K) -> Option<&mut V>;
}

/*
 ********* implementations *********
*/
impl<K, S, D> Params<K, ArrayBase<S, D>> for HashMap<K, ArrayBase<S, D>>
where
    S: Data,
    D: Dimension,
    K: core::cmp::Eq + core::hash::Hash,
{
    fn get(&self, param: &K) -> Option<&ArrayBase<S, D>> {
        self.get(param)
    }

    fn get_mut(&mut self, param: &K) -> Option<&mut ArrayBase<S, D>> {
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
