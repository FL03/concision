/*
    Appellation: optimize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # optimize
//!
pub use self::{optimizer::*, utils::*};

pub(crate) mod optimizer;

pub trait Optimize {
    // fn params(&self) -> &Params;
    fn optimize(&self) -> Self;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
