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
use crate::{HEADS, MODEL_SIZE, QUERY_SIZE};
use ndarray::prelude::{Dimension, Ix2, Ix3, Ix4};
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

pub trait Batched {
    fn batch(&self) -> usize;
}

impl Batched for Ix3 {
    fn batch(&self) -> usize {
        self[0]
    }
}

impl Batched for Ix4 {
    fn batch(&self) -> usize {
        self[0]
    }
}

pub enum Shapes {
    Batched((usize, Box<Shapes>)),
    Data(BaseShape),
    Head(HeadShape),
    Mask { seq: usize },
    MultiHead(MultiShape),
}

impl Shapes {
    pub fn batched(batch_size: usize, shape: impl Into<Shapes>) -> Self {
        Shapes::Batched((batch_size, Box::new(shape.into())))
    }
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
        Self::new(batch, seq, MODEL_SIZE)
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn model_size(&self) -> usize {
        self.model
    }

    pub fn seq_len(&self) -> usize {
        self.seq
    }
}

impl IntoDimension for BaseShape {
    type Dim = ndarray::Ix3;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix3(self.batch, self.seq, self.model)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct MultiShape {
    pub heads: usize,
    pub seq: usize,
    pub query: usize,
}

impl MultiShape {
    pub fn new(heads: usize, seq: usize, query: usize) -> Self {
        Self { heads, seq, query }
    }

    pub fn std(seq: usize) -> Self {
        Self::new(HEADS, seq, *QUERY_SIZE)
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn model_size(&self) -> usize {
        self.heads() * self.query_size()
    }

    pub fn query_size(&self) -> usize {
        self.query
    }

    pub fn seq_len(&self) -> usize {
        self.seq
    }
}

impl From<MultiShape> for Shapes {
    fn from(shape: MultiShape) -> Self {
        Shapes::MultiHead(shape)
    }
}

impl From<MultiShape> for [usize; 3] {
    fn from(shape: MultiShape) -> Self {
        [shape.heads, shape.seq, shape.query]
    }
}

impl From<MultiShape> for (usize, usize, usize) {
    fn from(shape: MultiShape) -> Self {
        (shape.heads, shape.seq, shape.query)
    }
}

impl IntoDimension for MultiShape {
    type Dim = ndarray::Ix3;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix3(self.heads, self.seq, self.query)
    }
}

pub trait HeadDimension {
    fn query_size(&self) -> usize;
    fn sequence(&self) -> usize;

    fn as_shape(&self) -> HeadShape {
        HeadShape::new(self.query_size(), self.sequence())
    }
}

pub trait AsShape<Sh> {
    fn as_shape(&self) -> Sh;
}

pub trait ShapeExt<D>
where
    D: ndarray::Dimension,
{
    fn dim(&self) -> D;
}

impl<T> HeadDimension for T
where
    T: Clone + IntoDimension<Dim = Ix2>,
{
    fn query_size(&self) -> usize {
        self.clone().into_dimension()[1]
    }

    fn sequence(&self) -> usize {
        self.clone().into_dimension()[0]
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

impl From<HeadShape> for (usize, usize) {
    fn from(shape: HeadShape) -> Self {
        (shape.seq, shape.query)
    }
}

impl From<[usize; 2]> for HeadShape {
    fn from(dim: [usize; 2]) -> Self {
        Self::new(dim[1], dim[0])
    }
}

impl From<HeadShape> for [usize; 2] {
    fn from(shape: HeadShape) -> Self {
        [shape.seq, shape.query]
    }
}

impl IntoDimension for HeadShape {
    type Dim = ndarray::Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.seq, self.query)
    }
}
