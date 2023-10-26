/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention Parameters
//!
//! ## Hyperparameters
pub use self::{dim::*, utils::*};

pub(crate) mod dim;

use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]

pub struct AttentionParams {
    pub batch: usize,
    pub dim: AttentionDim,
    pub heads: usize,
    pub samples: usize,
}

pub(crate) mod utils {}
