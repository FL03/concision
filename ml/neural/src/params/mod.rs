/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{bias::*, shapes::*, utils::*, weight::*};

pub(crate) mod bias;
pub(crate) mod shapes;
pub(crate) mod weight;

use crate::core::prelude::Borrowed;

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array2, Ix2, NdFloat};
use ndarray::Dimension;
use num::Float;

pub trait WeightTensor<T = f64, D = Ix2>
where
    Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
    D: Dimension,
    T: NdFloat,
{
    fn weights(&self) -> &Array<T, D>;
    fn weights_mut(&mut self) -> &mut Array<T, D>;
}

pub trait Parameter {}

pub trait Biased<T = f64>
where
    T: Float,
{
    fn bias(&self) -> &Bias<T>;
    fn bias_mut(&mut self) -> &mut Bias<T>;
}

impl<T, D> WeightTensor<T, D> for Array<T, D>
where
    Array<T, D>: Borrowed<Array<T, D>> + Dot<Array<T, D>, Output = Array<T, D>>,
    D: Dimension,
    T: NdFloat,
{
    fn weights(&self) -> &Array<T, D> {
        self.as_ref()
    }

    fn weights_mut(&mut self) -> &mut Array<T, D> {
        self.as_mut()
    }
}

pub trait Weighted<T = f64>
where
    T: Float,
{
    fn weights(&self) -> &Array2<T>;

    fn weights_mut(&mut self) -> &mut Array2<T>;
}

impl<S, T> Weighted<T> for S
where
    S: AsMut<Array2<T>> + AsRef<Array2<T>>,
    T: Float,
{
    fn weights(&self) -> &Array2<T> {
        self.as_ref()
    }

    fn weights_mut(&mut self) -> &mut Array2<T> {
        self.as_mut()
    }
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
