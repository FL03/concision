/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{HEADS, MODEL};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct MultiHeadParams {
    pub heads: usize,
    pub model: usize,
}

impl MultiHeadParams {
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

impl Default for MultiHeadParams {
    fn default() -> Self {
        Self::new(HEADS, MODEL)
    }
}

impl From<MultiHeadParams> for (usize, usize) {
    fn from(params: MultiHeadParams) -> Self {
        (params.heads, params.model)
    }
}

impl From<MultiHeadParams> for [usize; 2] {
    fn from(params: MultiHeadParams) -> Self {
        [params.heads, params.model]
    }
}
