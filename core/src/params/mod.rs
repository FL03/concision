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

pub trait Param {
    fn kind(&self) -> &ParamKind;

    fn name(&self) -> &str;
}

pub trait Biased<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, D>);
}

pub trait Weighted<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, D>);
}

pub trait Params<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the parameters of the layer.
    fn params(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the parameters of the layer.
    fn params_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the parameters of the layer.
    fn set_params(&mut self, params: Array<T, D>);
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
