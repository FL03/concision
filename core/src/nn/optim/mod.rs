/*
    Appellation: optim <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Optimizers
//!
//! This module contains various optimizers used for training neural networks.
pub use self::optimizer::*;

pub(crate) mod optimizer;

pub(crate) mod prelude {
    pub use super::Optimize;
    pub use super::optimizer::*;
}

pub trait Optimize {}
