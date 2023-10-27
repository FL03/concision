/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::params::{HeadShape, QKV};
use super::{AttentionSpace, HeadSpace};
use crate::neural::neurons::activate::{Activator, Softmax};
use ndarray::{Array2, IntoDimension};
use serde::{Deserialize, Serialize};
use std::ops;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct AttentionHead {
    dim: HeadShape,
    mask: Array2<f64>,
    
    score: Array2<f64>,
    weights: AttentionSpace,
}

impl AttentionHead {
    pub fn new(dim: HeadShape) -> Self {
        Self {
            dim,
            mask: Array2::zeros((dim.sequence(), dim.sequence())),
            score: Array2::zeros(dim.into_dimension()),
            weights: AttentionSpace::new(dim),
        }
    }

    pub fn attention(&mut self, data: &Array2<f64>) -> Array2<f64>{
        // multiply the data by the wieghted query, key, and value matrices, respectively
        let weighted = self.weights.clone() * data;
        let (q, k, v) = weighted.qkv();

        // compute the attention score
        let score = {
            let inner = (q.dot(&k.t()) + self.mask.clone()) * self.scale();
            Softmax::rho(inner).dot(&v)
        };
        self.score = score.clone();
        score
    }

    pub fn dim(&self) -> HeadShape {
        self.dim
    }

    pub fn mask(&self) -> &Array2<f64> {
        &self.mask
    }

    pub fn mask_mut(&mut self) -> &mut Array2<f64> {
        &mut self.mask
    }

    pub fn process(&mut self, data: &Array2<f64>) {
        // multiply the data by the wieghted query, key, and value matrices, respectively
        let weighted = self.weights.clone() * data;
        let (q, k, v) = weighted.qkv();

        // compute the attention score
        self.score = {
            let inner = (q.dot(&k.t()) + self.mask.clone()) * self.scale();
            Softmax::rho(inner).dot(&v)
        };
    }

    pub fn scale(&self) -> f64 {
        1.0 / (self.dim.query_size() as f64).sqrt()
    }

    pub fn score(&self) -> &Array2<f64> {
        &self.score
    }

    pub fn set_mask(&mut self, mask: Array2<f64>) {
        self.mask = mask;
    }

    pub fn with_mask(mut self, mask: Array2<f64>) -> Self {
        self.mask = mask;
        self
    }
}

impl HeadSpace for AttentionHead {
    fn query(&self) -> &Array2<f64> {
        &self.weights.query
    }

    fn key(&self) -> &Array2<f64> {
        &self.weights.key
    }

    fn value(&self) -> &Array2<f64> {
        &self.weights.value
    }
}

impl std::fmt::Display for AttentionHead {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl ops::Index<QKV> for AttentionHead {
    type Output = Array2<f64>;

    fn index(&self, index: QKV) -> &Self::Output {
        &self.weights[index]
    }
}

impl ops::IndexMut<QKV> for AttentionHead {
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        &mut self.weights[index]
    }
}

impl ops::Mul<Array2<f64>> for AttentionHead {
    type Output = AttentionHead;

    fn mul(self, rhs: Array2<f64>) -> Self::Output {
        let mut head = self.clone();
        head.weights = self.weights * rhs;
        head
    }
}

impl ops::Mul<&Array2<f64>> for AttentionHead {
    type Output = AttentionHead;

    fn mul(self, rhs: &Array2<f64>) -> Self::Output {
        let mut head = self.clone();
        head.weights = self.weights * rhs;
        head
    }
}

impl ops::MulAssign<Array2<f64>> for AttentionHead {
    fn mul_assign(&mut self, rhs: Array2<f64>) {
        self.weights *= rhs;
    }
}

impl ops::MulAssign<&Array2<f64>> for AttentionHead {
    fn mul_assign(&mut self, rhs: &Array2<f64>) {
        self.weights *= rhs;
    }
}

impl ops::MulAssign<&Array2<f64>> for &mut AttentionHead {
    fn mul_assign(&mut self, rhs: &Array2<f64>) {
        self.weights *= rhs;
    }
}
