/*
    Appellation: cost <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # cost
//!
pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

pub trait Cost {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
