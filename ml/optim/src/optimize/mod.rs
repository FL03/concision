/*
    Appellation: optimize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # optimize
//!
pub use self::{optimizer::*, utils::*};

pub(crate) mod optimizer;

pub trait Optimize {
    type Model;

    fn apply(&self, model: &mut Self::Model) -> &mut Self::Model;

    fn model(&self) -> &Self::Model;

    fn model_mut(&mut self) -> &mut Self::Model;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
