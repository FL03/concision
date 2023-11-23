/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::params::{HeadShape, QKV};
use super::Weight;
use crate::neural::func::activate::{Activate, Softmax};
use ndarray::prelude::{Array2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use std::ops;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct AttentionHead<T: Float> {
    dim: HeadShape,
    mask: Array2<T>,
    weights: Weight<T>,
}

impl<T> AttentionHead<T>
where
    T: Float,
{
    pub fn dim(&self) -> HeadShape {
        self.dim
    }

    pub fn mask_mut(&mut self) -> &mut Array2<T> {
        &mut self.mask
    }

    pub fn scale(&self) -> T {
        T::one() / T::from(self.dim.query_size()).unwrap().sqrt()
    }

    pub fn weights(&self) -> &Weight<T> {
        &self.weights
    }

    pub fn set_mask(&mut self, mask: Array2<T>) {
        self.mask = mask;
    }

    pub fn with_mask(mut self, mask: Array2<T>) -> Self {
        self.mask = mask;
        self
    }
}

impl<T> AttentionHead<T>
where
    T: Float + SampleUniform,
{
    pub fn new(dim: HeadShape) -> Self {
        Self {
            dim,
            mask: Array2::zeros((dim.sequence(), dim.sequence())),
            weights: Weight::uniform(dim),
        }
    }
}

impl<T> AttentionHead<T>
where
    T: NdFloat,
{
    pub fn attention(&mut self, data: &Array2<T>) -> Array2<T> {
        // multiply the data by the wieghted query, key, and value matrices, respectively
        let weighted = data * self.weights();
        let (q, k, v) = weighted.qkv();

        // compute the attention score
        let inner = (q.dot(&k.t()) + self.mask.clone()) * self.scale();
        Softmax::default().activate(&inner).dot(&v)
    }
}

impl<T> std::fmt::Display for AttentionHead<T>
where
    T: Float + Serialize,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<T> ops::Index<QKV> for AttentionHead<T>
where
    T: Float,
{
    type Output = Array2<T>;

    fn index(&self, index: QKV) -> &Self::Output {
        &self.weights[index]
    }
}

impl<T> ops::IndexMut<QKV> for AttentionHead<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        &mut self.weights[index]
    }
}

impl<T> ops::Mul<Array2<T>> for AttentionHead<T>
where
    T: NdFloat,
{
    type Output = AttentionHead<T>;

    fn mul(self, rhs: Array2<T>) -> Self::Output {
        let mut head = self.clone();
        head.weights = self.weights * rhs;
        head
    }
}

impl<T> ops::Mul<&Array2<T>> for AttentionHead<T>
where
    T: NdFloat,
{
    type Output = AttentionHead<T>;

    fn mul(self, rhs: &Array2<T>) -> Self::Output {
        let mut head = self.clone();
        head.weights = self.weights * rhs;
        head
    }
}

impl<T> ops::MulAssign<Array2<T>> for AttentionHead<T>
where
    T: NdFloat,
{
    fn mul_assign(&mut self, rhs: Array2<T>) {
        self.weights *= rhs;
    }
}

impl<T> ops::MulAssign<&Array2<T>> for AttentionHead<T>
where
    T: NdFloat,
{
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        self.weights *= rhs;
    }
}

impl<T> ops::MulAssign<&Array2<T>> for &mut AttentionHead<T>
where
    T: NdFloat,
{
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        self.weights *= rhs;
    }
}
