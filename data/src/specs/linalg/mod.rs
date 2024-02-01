/*
   Appellation: linalg <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Algebra
//!
//! This module implements a number of linear algebra operations.
pub use self::matmul::*;

pub(crate) mod matmul;

#[cfg(test)]
mod tests {}
