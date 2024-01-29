/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Structured State Space Sequence Model (S4)
//!
//! ## Overview
//!
//! ## References
//!     - [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
pub use self::{config::*, model::*, params::*, state::*};

pub(crate) mod config;
pub(crate) mod model;
pub(crate) mod params;
pub(crate) mod state;

#[cfg(test)]
mod tests {}
