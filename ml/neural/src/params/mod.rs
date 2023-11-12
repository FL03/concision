/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{bias::*, utils::*, weight::*};

pub(crate) mod bias;
pub(crate) mod weight;

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array2, Ix2};
use ndarray::Dimension;
use num::Float;

pub enum ParameterShapes {
    Thick { features: usize, outputs: usize },
}

pub trait Parameter {}

pub trait Biased<T = f64>
where
    T: Float,
{
    fn bias(&self) -> &Bias<T>;
    fn bias_mut(&mut self) -> &mut Bias<T>;
}

pub trait W<T = f64, D = Ix2>
where
    Self: Dot<Array<T, D>>,
    D: Dimension,
    T: Float,
{
    fn weights(&self) -> &Self;
    fn weights_mut(&mut self) -> &Self;
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
