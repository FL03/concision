/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{neuron::*, node::*, utils::*};

pub(crate) mod neuron;
pub(crate) mod node;

pub mod activate;

pub trait Weight {}

pub(crate) mod utils {}
