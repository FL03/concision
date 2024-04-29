/*
   Appellation: compose <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Compose
//!
//! This module implements traits enabling the dynamic generation of data.
pub use self::{arange::*, generate::*};

pub(crate) mod arange;
pub(crate) mod generate;

#[cfg(test)]
mod tests {}
