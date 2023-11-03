/*
   Appellation: weights <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Weights
pub use self::{weight::*, utils::*};

pub(crate) mod weight;


use ndarray::prelude::Array2;
use num::Float;

pub trait Weighted<T = f64> where T: Float {
    fn weights(&self) -> &Array2<T>;

    fn weights_mut(&mut self) -> &mut Array2<T>;
}

impl<S, T> Weighted<T> for S where S: AsMut<Array2<T>> + AsRef<Array2<T>>, T: Float {
    fn weights(&self) -> &Array2<T> {
        self.as_ref()
    }

    fn weights_mut(&mut self) -> &mut Array2<T> {
        self.as_mut()
    }
}


pub(crate) mod utils {
}

#[cfg(test)]
mod tests {}
