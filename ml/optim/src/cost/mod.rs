/*
    Appellation: cost <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # cost
//!
pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

use ndarray::prelude::{Array, Array1};
use ndarray::Dimension;
use num::Float;

pub trait Cost<T = f64>
where
    T: Float,
{
    fn cost(&self, other: &Self) -> T;
}

pub trait CostArr<T = f64>
where
    T: Float,
{
    type Dim: Dimension;

    fn cost(&self, pred: &Array<T, Self::Dim>, target: &Array1<T>) -> Array<T, Self::Dim>;
}

pub(crate) mod utils {

    use ndarray::prelude::Array;
    use ndarray::Dimension;
    use num::{Float, FromPrimitive};

    pub fn mse<'a, T, D>(pred: &Array<T, D>, target: &Array<T, D>) -> T
    where
        T: Float + FromPrimitive,
        D: Dimension,
    {
        (pred - target)
            .mapv(|x| x.powi(2))
            .mean()
            .unwrap_or_else(T::zero)
    }

    pub fn mae<'a, T, D>(pred: &Array<T, D>, target: &Array<T, D>) -> Array<T, D>
    where
        T: Float,
        D: Dimension,
    {
        (pred - target).mapv(|x| x.abs())
    }
}

#[cfg(test)]
mod tests {}
