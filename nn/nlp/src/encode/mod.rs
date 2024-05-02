/*
   Appellation: encode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub mod positional;

use ndarray::prelude::{Array, Array2};
use ndarray::Dimension;

pub trait Encode<T> {
    type Output;

    fn encode(&self, data: &T) -> Self::Output;
}

pub trait EncodeArr<T> {
    type Dim: Dimension;

    fn encode(&self, data: &Array<T, Self::Dim>) -> Array2<T>;
}

#[cfg(test)]
mod tests {}
