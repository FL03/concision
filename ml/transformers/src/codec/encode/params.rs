/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{HEADS, MODEL_SIZE};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct EncoderParams {
    pub heads: usize,
    pub model: usize,
}

impl EncoderParams {
    pub fn new(heads: usize, model: usize) -> Self {
        Self { heads, model }
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn model_size(&self) -> usize {
        self.model
    }

    pub fn query_size(&self) -> usize {
        self.model / self.heads
    }
}

impl Default for EncoderParams {
    fn default() -> Self {
        Self::new(HEADS, MODEL_SIZE)
    }
}
