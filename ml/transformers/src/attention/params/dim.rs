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
use crate::{DEFAULT_ATTENTION_HEADS, DEFAULT_EMBEDDING_SIZE};
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

pub trait StructuredDim: IntoDimension {}

pub trait BaseDimension: IntoDimension {
    fn batch_size(&self) -> usize;
    fn model_size(&self) -> usize;
    fn seq_len(&self) -> usize;
}

pub trait MultiHeadDimension: BaseDimension {
    fn heads(&self) -> usize;
}

impl<D> BaseDimension for D
where
    D: Clone + IntoDimension,
{
    fn batch_size(&self) -> usize {
        self.clone().into_dimension()[0]
    }

    fn model_size(&self) -> usize {
        self.clone().into_dimension()[2]
    }

    fn seq_len(&self) -> usize {
        self.clone().into_dimension()[1]
    }
}

pub enum AttentionDims {
    Base(BaseShape),       // a 3d matrix (batch, seq, model)
    Head(HeadShape),       // a 2d matrix (seq, query)
    MultiHead(MultiShape), // a 4d matrix (batch, heads, seq, query)
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct BaseShape {
    pub batch: usize,
    pub seq: usize,
    pub model: usize,
}

impl BaseShape {
    pub fn new(batch: usize, seq: usize, model: usize) -> Self {
        Self { batch, seq, model }
    }

    pub fn std(batch: usize, seq: usize) -> Self {
        Self::new(batch, seq, DEFAULT_EMBEDDING_SIZE)
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn model_size(&self) -> usize {
        self.model
    }

    pub fn seq_len(&self) -> usize {
        self.seq
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct MultiShape {
    pub batch: usize,
    pub heads: usize,
    pub seq: usize,
    pub query: usize,
}

impl MultiShape {
    pub fn new(batch: usize, heads: usize, seq: usize, query: usize) -> Self {
        Self {
            batch,
            heads,
            seq,
            query,
        }
    }

    pub fn std(batch: usize, seq: usize) -> Self {
        Self::new(batch, DEFAULT_ATTENTION_HEADS, seq, DEFAULT_EMBEDDING_SIZE)
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn model_size(&self) -> usize {
        self.query
    }

    pub fn seq_len(&self) -> usize {
        self.seq
    }
}

///
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct HeadShape {
    pub query: usize, // cols
    pub seq: usize,   // rows
}

impl HeadShape {
    pub fn new(query: usize, seq: usize) -> Self {
        Self { query, seq }
    }

    pub fn query_size(&self) -> usize {
        self.query
    }

    pub fn sequence(&self) -> usize {
        self.seq
    }

    pub fn scale(&self) -> f64 {
        1.0 / (self.query as f64).sqrt()
    }
}

impl From<(usize, usize)> for HeadShape {
    fn from((seq, query): (usize, usize)) -> Self {
        Self::new(query, seq)
    }
}

impl From<[usize; 2]> for HeadShape {
    fn from(dim: [usize; 2]) -> Self {
        Self::new(dim[1], dim[0])
    }
}

impl From<HeadShape> for (usize, usize) {
    fn from(shape: HeadShape) -> Self {
        (shape.seq, shape.query)
    }
}

impl IntoDimension for HeadShape {
    type Dim = ndarray::Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.seq, self.query)
    }
}
