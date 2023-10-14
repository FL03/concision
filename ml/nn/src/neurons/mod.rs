/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{neuron::*, utils::*};

pub(crate) mod neuron;

pub mod activate;

pub trait Weight {}

pub(crate) mod utils {}