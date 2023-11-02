/*
    Appellation: arch <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Architecture
//!
//! This module describes the architecture of various components of the neural network.
pub use self::{architecture::*, utils::*};

pub(crate) mod architecture;

pub trait Arch {}

pub(crate) mod utils {}
