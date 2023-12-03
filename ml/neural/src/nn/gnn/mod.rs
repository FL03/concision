/*
    Appellation: graph <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Graph Neural Network
//!
pub use self::{model::*, tasks::*, utils::*};

pub(crate) mod model;
pub(crate) mod tasks;

use num::Float;

pub trait GNN<T = f64>
where
    T: Float,
{
    type G;

    fn depth(&self) -> usize {
        self.layers().len()
    }

    fn layers(&self) -> &[Self::G];
}

pub(crate) mod utils {}

#[cfg(tets)]
mod tests {}
