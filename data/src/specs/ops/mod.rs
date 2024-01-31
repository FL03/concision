/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Operations
//!
//! A matrix is a two-dimensional array of elements.
pub use self::{arange::*, arithmetic::*};

pub(crate) mod arange;
pub(crate) mod arithmetic;

pub trait Truncate {
    fn trunc(self) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_arange() {
        let exp = array![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(&exp, &Array1::<f64>::arange(5))
    }
}
