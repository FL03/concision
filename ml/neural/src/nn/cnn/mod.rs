/*
    Appellation: cnn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concurrent Neural Network (CNN)
//!
//!  
pub use self::{model::*, utils::*};

pub(crate) mod model;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
