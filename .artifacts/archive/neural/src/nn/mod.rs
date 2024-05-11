/*
    Appellation: nn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Neural Network
pub use self::{kinds::*, position::*};

pub(crate) mod kinds;
pub(crate) mod position;

pub mod gnn;

pub trait NeuralNetwork {}
