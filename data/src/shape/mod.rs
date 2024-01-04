/*
   Appellation: shapes <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Shapes
pub use self::{dimension::*, rank::*, shape::*};

pub(crate) mod dimension;
pub(crate) mod rank;
pub(crate) mod shape;

#[cfg(test)]
mod tests {}
