/*
    Appellation: graph <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Graph Neural Network
//!
pub use self::{model::*, utils::*};

pub(crate) mod model;

pub(crate) mod utils {}

#[cfg(tets)]
mod tests {}
