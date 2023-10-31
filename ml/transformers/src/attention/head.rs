/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::params::{HeadShape, QKV};
use super::{Head, Weight};
use crate::neural::neurons::activate::{Activator, Softmax};
use ndarray::prelude::Array2;
use ndarray::ScalarOperand;
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

impl<T: Float + ScalarOperand> AttentionHead<T> {
    pub fn new(dim: HeadShape) -> Self {
        Self {
            dim,
            mask: Array2::zeros((dim.sequence(), dim.sequence())),
            weights: Weight::new(dim),
        }
    }

    pub fn attention(&mut self, data: &Array2<T>) -> Array2<T> {
        // multiply the data by the wieghted query, key, and value matrices, respectively
        let weighted = self.weights.clone() * data;
        let (q, k, v) = weighted.qkv();

        // compute the attention score
        let inner = (q.dot(&k.t()) + self.mask.clone()) * self.scale();
        Softmax::rho(inner).dot(&v)
    }

    pub fn dim(&self) -> HeadShape {
        self.dim
    }

    pub fn mask(&self) -> &Array2<T> {
        &self.mask
    }

    pub fn mask_mut(&mut self) -> &mut Array2<T> {
        &mut self.mask
    }

    pub fn scale(&self) -> T {
        T::one() / T::from(self.dim.query_size()).unwrap().sqrt()
    }

    pub fn set_mask(&mut self, mask: Array2<T>) {
        self.mask = mask;
    }

    pub fn with_mask(mut self, mask: Array2<T>) -> Self {
        self.mask = mask;
        self
    }
}

impl<T: Float> Head<T> for AttentionHead<T> {
    fn query(&self) -> &Array2<T> {
        &self.weights.query
    }

    fn key(&self) -> &Array2<T> {
        &self.weights.key
    }

    fn value(&self) -> &Array2<T> {
        &self.weights.value
    }
}

impl<T: Float + Serialize> std::fmt::Display for AttentionHead<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<T: Float> ops::Index<QKV> for AttentionHead<T> {
    type Output = Array2<T>;

    fn index(&self, index: QKV) -> &Self::Output {
        &self.weights[index]
    }
}

impl<T: Float> ops::IndexMut<QKV> for AttentionHead<T> {
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        &mut self.weights[index]
    }
}

impl<T: Float + 'static> ops::Mul<Array2<T>> for AttentionHead<T> {
    type Output = AttentionHead<T>;

    fn mul(self, rhs: Array2<T>) -> Self::Output {
        let mut head = self.clone();
        head.weights = self.weights * rhs;
        head
    }
}

impl<T: Float + 'static> ops::Mul<&Array2<T>> for AttentionHead<T> {
    type Output = AttentionHead<T>;

    fn mul(self, rhs: &Array2<T>) -> Self::Output {
        let mut head = self.clone();
        head.weights = self.weights * rhs;
        head
    }
}

impl<T: Float + 'static> ops::MulAssign<Array2<T>> for AttentionHead<T> {
    fn mul_assign(&mut self, rhs: Array2<T>) {
        self.weights *= rhs;
    }
}

impl<T: Float + 'static> ops::MulAssign<&Array2<T>> for AttentionHead<T> {
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        self.weights *= rhs;
    }
}

impl<T: Float + 'static> ops::MulAssign<&Array2<T>> for &mut AttentionHead<T> {
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        self.weights *= rhs;
    }
}
