/*
   Appellation: dim <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Dimensions
//!
//! The dimensionality of the attention mechanism is defined by the following:
//!
//! - `attention`: The dimension of the key, query, and value vectors
//! - `batch`: The batch size
//! - `heads`: The number of attention heads
//! - `model`: The dimension of the model (embedding size)
use crate::{DEFAULT_ATTENTION_HEADS, DEFAULT_EMBEDDING_SIZE, DEFAULT_SAMPLE_SIZE};
use serde::{Deserialize, Serialize};

pub enum AttentionDims {
    Head {
        query: usize,
        seq: usize,
    },
    Context {
        batch: usize,
        heads: usize,
        model: usize,
        seq: usize,
        samples: usize,
    },
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct AttentionDim {
    pub batch: usize,   // The batch size
    pub heads: usize,   // The number of attention heads
    pub model: usize,   // The dimension of the model (embedding size)
    pub seq: usize,     // The sequence length
    pub samples: usize, // The number of samples
}

impl AttentionDim {
    pub fn new(batch: usize, heads: usize, model: usize, seq: usize, samples: usize) -> Self {
        Self {
            batch,
            heads,
            model,
            seq,
            samples,
        }
    }

    pub fn std(batch: usize, seq: usize) -> Self {
        Self::new(
            batch,
            DEFAULT_ATTENTION_HEADS,
            DEFAULT_EMBEDDING_SIZE,
            seq,
            DEFAULT_SAMPLE_SIZE,
        )
    }

    // The dimension of the key, query, and value vectors
    pub fn query_size(&self) -> usize {
        self.model / self.heads
    }

    pub fn head_dim(&self) -> (usize, usize) {
        (self.seq, self.query_size())
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct HeadShape {
    pub query: usize,
    pub seq: usize,
}

impl HeadShape {
    pub fn new(query: usize, seq: usize) -> Self {
        Self { query, seq }
    }

    pub fn scale(&self) -> f64 {
        1.0 / (self.query as f64).sqrt()
    }
}
