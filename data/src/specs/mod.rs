/*
   Appellation: specs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{elements::*, export::*, import::*};

pub(crate) mod elements;
pub(crate) mod export;
pub(crate) mod import;

use ndarray::prelude::{Array1, Array2};

pub trait Records<T> {
    fn features(&self) -> usize;

    fn samples(&self) -> usize;
}

impl<T> Records<T> for Array1<T> {
    fn features(&self) -> usize {
        1
    }

    fn samples(&self) -> usize {
        self.shape()[0]
    }
}

impl<T> Records<T> for Array2<T> {
    fn features(&self) -> usize {
        self.shape()[1]
    }

    fn samples(&self) -> usize {
        self.shape()[0]
    }
}

#[cfg(test)]
mod tests {}
