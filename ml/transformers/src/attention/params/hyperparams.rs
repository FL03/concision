/*
   Appellation: hyperparams <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Hyperparameters
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

use super::dim::{BaseShape, HeadShape, MultiShape};
use crate::{HEADS, MODEL_SIZE, SAMPLES};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]

pub struct AttentionParameters {
    pub batch: usize,
    pub heads: usize,
    pub model: usize,
    pub samples: usize,
    pub seq: usize,
}

impl AttentionParameters {
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
        Self::new(batch, HEADS, MODEL_SIZE, SAMPLES, seq)
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

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn with_heads(mut self, heads: usize) -> Self {
        self.heads = heads;
        self
    }

    pub fn with_model(mut self, model: usize) -> Self {
        self.model = model;
        self
    }

    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    pub fn with_seq(mut self, seq: usize) -> Self {
        self.seq = seq;
        self
    }
}

impl From<AttentionParameters> for BaseShape {
    fn from(params: AttentionParameters) -> Self {
        Self::new(params.batch, params.seq, params.model)
    }
}

impl From<AttentionParameters> for MultiShape {
    fn from(params: AttentionParameters) -> Self {
        Self::new(params.heads, params.seq, params.model)
    }
}

impl From<AttentionParameters> for HeadShape {
    fn from(params: AttentionParameters) -> Self {
        Self::new(params.seq, params.query_size())
    }
}
