/*
   Appellation: mat <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Matrix
//!
//! A matrix is a two-dimensional array of elements.
pub use self::matrix::*;

pub(crate) mod matrix;

pub trait Mat<T = f64> {}

#[cfg(test)]
mod tests {}
