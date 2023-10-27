/*
   Appellation: context <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::params::QKV;
use ndarray::{Array2, IntoDimension, Ix2};
use serde::{Deserialize, Serialize};
use std::ops;

pub trait LinearLayer {
    fn matmul(&self, data: &Array2<f64>) -> Array2<f64>;
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AttentionSpace {
    dim: Ix2,
    pub key: Array2<f64>,
    pub query: Array2<f64>,
    pub value: Array2<f64>,
}

impl AttentionSpace {
    pub fn new<D>(dim: D) -> Self
    where
        D: IntoDimension<Dim = Ix2>,
    {
        let dim = dim.into_dimension();
        Self {
            dim,
            key: Array2::ones(dim),
            query: Array2::ones(dim),
            value: Array2::ones(dim),
        }
    }

    pub fn dim(&self) -> Ix2 {
        self.dim
    }

    pub fn matmul(&self) -> Array2<f64> {
        self.query.dot(&self.key.t())
    }

    pub fn qkv(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        (self.query.clone(), self.key.clone(), self.value.clone())
    }

    pub fn set_weight(&mut self, qkv: QKV, weight: Array2<f64>) {
        match qkv {
            QKV::Key => self.key = weight,
            QKV::Query => self.query = weight,
            QKV::Value => self.value = weight,
        }
    }

    pub fn set_weights(&mut self, qkv: impl IntoIterator<Item = QKV>, weight: Array2<f64>) {
        for qkv in qkv {
            self.set_weight(qkv, weight.clone());
        }
    }

    pub fn with_weight(mut self, query: &Array2<f64>, key: &Array2<f64>, value: &Array2<f64>) {
        self.key = key.clone();
        self.query = query.clone();
        self.value = value.clone();
    }
}

impl std::fmt::Display for AttentionSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<D> From<D> for AttentionSpace
where
    D: IntoDimension<Dim = Ix2>,
{
    fn from(dim: D) -> Self {
        let dim = dim.into_dimension();
        let arr = Array2::ones(dim);
        Self {
            dim,
            key: arr.clone(),
            query: arr.clone(),
            value: arr,
        }
    }
}

impl From<AttentionSpace> for (Array2<f64>, Array2<f64>, Array2<f64>) {
    fn from(context: AttentionSpace) -> Self {
        (context.key, context.query, context.value)
    }
}

impl ops::Index<QKV> for AttentionSpace {
    type Output = Array2<f64>;

    fn index(&self, index: QKV) -> &Self::Output {
        use QKV::*;
        match index {
            Key => &self.key,
            Query => &self.query,
            Value => &self.value,
        }
    }
}

impl ops::IndexMut<QKV> for AttentionSpace {
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        match index {
            QKV::Key => &mut self.key,
            QKV::Query => &mut self.query,
            QKV::Value => &mut self.value,
        }
    }
}

impl ops::Mul<Array2<f64>> for AttentionSpace {
    type Output = Self;

    fn mul(self, rhs: Array2<f64>) -> Self::Output {
        let mut ctx = self.clone();
        ctx.key = ctx.key.dot(&rhs);
        ctx.query = ctx.query.dot(&rhs);
        ctx.value = ctx.value.dot(&rhs);
        ctx
    }
}

impl ops::Mul<&Array2<f64>> for AttentionSpace {
    type Output = Self;

    fn mul(self, rhs: &Array2<f64>) -> Self::Output {
        let mut ctx = self.clone();
        ctx.key = ctx.key.dot(rhs);
        ctx.query = ctx.query.dot(rhs);
        ctx.value = ctx.value.dot(rhs);
        ctx
    }
}

impl ops::Mul<&Array2<f64>> for &AttentionSpace {
    type Output = AttentionSpace;

    fn mul(self, rhs: &Array2<f64>) -> Self::Output {
        let mut ctx = self.clone();
        ctx.key = ctx.key.dot(rhs);
        ctx.query = ctx.query.dot(rhs);
        ctx.value = ctx.value.dot(rhs);
        ctx
    }
}

impl ops::MulAssign<Array2<f64>> for AttentionSpace {
    fn mul_assign(&mut self, rhs: Array2<f64>) {
        self.key = self.key.dot(&rhs);
        self.query = self.query.dot(&rhs);
        self.value = self.value.dot(&rhs);
    }
}

impl ops::MulAssign<&Array2<f64>> for AttentionSpace {
    fn mul_assign(&mut self, rhs: &Array2<f64>) {
        self.key = self.key.dot(rhs);
        self.query = self.query.dot(rhs);
        self.value = self.value.dot(rhs);
    }
}

impl ops::MulAssign<&Array2<f64>> for &mut AttentionSpace {
    fn mul_assign(&mut self, rhs: &Array2<f64>) {
        self.key = self.key.dot(rhs);
        self.query = self.query.dot(rhs);
        self.value = self.value.dot(rhs);
    }
}
