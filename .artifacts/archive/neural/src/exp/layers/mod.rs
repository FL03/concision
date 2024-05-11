/*
    Appellation: exp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Experimental Layers
pub use self::{config::*, sublayer::*, wrapper::*};

pub(crate) mod config;
pub(crate) mod sublayer;
pub(crate) mod wrapper;

pub trait Layer {
    type Config;
    type Params;
}

#[cfg(test)]
mod tests {}
