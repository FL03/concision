/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention Parameters
//!
//! ### Hyperparameters
//!
//! Hyperparameters are one which are set before training and are not updated.
//!
//! The hyperparameters for the attention mechanism are:
//!    - batch: The number of samples in a batch.
//!    - heads: The number of attention heads.
//!    - model: The dimension of the model (embedding size).
//!    - samples: The number of samples to draw from the attention distribution.
//!
//!
pub use self::{dim::*, hyperparams::*, qkv::*, utils::*};

pub(crate) mod dim;
pub(crate) mod hyperparams;
pub(crate) mod qkv;

pub(crate) mod utils {}
