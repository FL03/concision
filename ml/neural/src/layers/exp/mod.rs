/*
    Appellation: exp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Experimental Layers
pub use self::{config::*, layer::*, sublayer::*, utils::*, wrapper::*};

pub(crate) mod config;
pub(crate) mod layer;
pub(crate) mod sublayer;
pub(crate) mod wrapper;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
