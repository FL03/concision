/*
    Appellation: cmp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{features::*, kinds::*, utils::*};

pub(crate) mod features;
pub(crate) mod kinds;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
