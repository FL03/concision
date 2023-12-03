/*
    Appellation: prop <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Propagation
//!
//! This module describes the propagation of data through a neural network.
pub use self::{modes::*, utils::*};

pub(crate) mod modes;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
