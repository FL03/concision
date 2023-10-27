/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention Parameters
//!
//! ## Hyperparameters
pub use self::{dim::*, qkv::*, utils::*};

pub(crate) mod dim;
pub(crate) mod qkv;

use crate::{DEFAULT_ATTENTION_HEADS, DEFAULT_EMBEDDING_SIZE, DEFAULT_SAMPLE_SIZE};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]

pub struct Hyperparameters {
    pub batch: usize,
    pub heads: usize,
    pub model: usize,
    pub samples: usize,
    pub seq: usize,
}

impl Hyperparameters {
    pub fn new(batch: usize, heads: usize, model: usize, samples: usize, seq: usize) -> Self {
        Self {
            batch,
            heads,
            model,
            samples,
            seq,
        }
    }

    pub fn std(batch: usize, seq: usize) -> Self {
        Self::new(
            batch,
            DEFAULT_ATTENTION_HEADS,
            DEFAULT_EMBEDDING_SIZE,
            DEFAULT_SAMPLE_SIZE,
            seq,
        )
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn model_size(&self) -> usize {
        self.model
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn seq_len(&self) -> usize {
        self.seq
    }

    pub fn query_size(&self) -> usize {
        self.model / self.heads
    }
}

impl From<Hyperparameters> for BaseShape {
    fn from(hyper: Hyperparameters) -> Self {
        Self::new(hyper.batch, hyper.seq, hyper.model)
    }
}

impl From<Hyperparameters> for MultiShape {
    fn from(hyper: Hyperparameters) -> Self {
        Self::new(hyper.batch, hyper.heads, hyper.seq, hyper.model)
    }
}

impl From<Hyperparameters> for HeadShape {
    fn from(hyper: Hyperparameters) -> Self {
        Self::new(hyper.seq, hyper.query_size())
    }
}

pub(crate) mod utils {}
