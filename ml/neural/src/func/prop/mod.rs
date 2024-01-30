/*
    Appellation: prop <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Propagation
//!
//! This module describes the propagation of data through a neural network.
pub use self::{modes::*, results::*};

pub(crate) mod modes;
pub(crate) mod results;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
