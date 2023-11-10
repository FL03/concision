/*
   Appellation: weights <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Weights
//!
//! ## Overview
//!
//! The `weights` module provides a `Weight` struct that is used to
//! group the `key`, `query`, and `value` matrices leveraged by the
//! attention mechanism.
//!
//! ## Dimensionality
//!
//! Each of the `key`, `query`, and `value` weight tensors are
//! initialized as square matrices (model, model)
//!
//!     - W(model, model)
//!     - Q/K/V(seq, model) * W(model, model) = (seq, model)
//!     - Split(Q/K/V) = (heads, seq, model/heads) = (heads, seq, query)
//!     - Q(seq, model) * Key(seq, model)^T = (seq, seq)
//!     - (seq, seq) + Mask(seq, seq) = (seq, seq)
//!     - (seq, seq) * V(seq, model) = (seq, model)
//!
//!
//!
use super::params::QKV;
use super::Weights;
use crate::core::GenerateRandom;
use crate::ops::Split;
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array2, Array3, Ix2};
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use std::ops;
use strum::IntoEnumIterator;

pub type WeightTensor<T = f64> = Array<T, Ix2>; // (seq, model)

pub enum AttentionTensor<T = f64> {
    Embedding(Array2<T>),
    Multihead(Array3<T>),
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Weight<T = f64>
where
    T: Float,
{
    dim: Ix2,
    pub key: Array2<T>,
    pub query: Array2<T>,
    pub value: Array2<T>,
}

impl<T: Float> Weight<T> {
    pub fn dim(&self) -> Ix2 {
        self.dim
    }

    pub fn qkv(&self) -> (Array2<T>, Array2<T>, Array2<T>) {
        self.clone().into()
    }
}

impl<T> Weight<T>
where
    T: Default + Float,
{
    pub fn new(dim: impl IntoDimension<Dim = Ix2>) -> Self {
        let dim = dim.into_dimension();
        let arr = Array2::default(dim);
        Self {
            dim,
            key: arr.clone(),
            query: arr.clone(),
            value: arr,
        }
    }
}

impl<T> Weight<T>
where
    T: Float + SampleUniform,
{
    pub fn uniform(dim: impl IntoDimension<Dim = Ix2>) -> Self {
        let dim = dim.into_dimension();
        Self {
            dim: dim.clone(),
            key: Array2::uniform(1, dim.clone()),
            query: Array2::uniform(1, dim.clone()),
            value: Array2::uniform(1, dim),
        }
    }
    pub fn init_uniform(mut self) -> Self {
        self.key = Array2::uniform(1, self.dim);
        self.query = Array2::uniform(1, self.dim);
        self.value = Array2::uniform(1, self.dim);
        self
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<T: Float> Split<(Array3<T>, Array3<T>, Array3<T>)> for Weight<T> {
    type Error = ndarray::ShapeError;

    fn split(&self, heads: usize) -> Result<(Array3<T>, Array3<T>, Array3<T>), Self::Error> {
        let (key, query, value) = self.qkv();
        Ok((key.split(heads)?, query.split(heads)?, value.split(heads)?))
    }
}

impl<T: Float + 'static> Weights<T> for Weight<T> {
    fn key(&self) -> &Array2<T> {
        &self.key
    }

    fn query(&self) -> &Array2<T> {
        &self.query
    }

    fn value(&self) -> &Array2<T> {
        &self.value
    }
}

impl<D, T> From<D> for Weight<T>
where
    D: IntoDimension<Dim = Ix2>,
    T: Float,
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

impl<T: Float> From<Weight<T>> for (Array2<T>, Array2<T>, Array2<T>) {
    fn from(context: Weight<T>) -> Self {
        (context.key, context.query, context.value)
    }
}

impl<T: Float + 'static> Dot<Array2<T>> for Weight<T> {
    type Output = Self;

    fn dot(&self, rhs: &Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = ctx[qkv].dot(rhs);
        }
        ctx
    }
}

impl<T: Float> ops::Index<QKV> for Weight<T> {
    type Output = Array2<T>;

    fn index(&self, index: QKV) -> &Self::Output {
        use QKV::*;
        match index {
            Key => &self.key,
            Query => &self.query,
            Value => &self.value,
        }
    }
}

impl<T: Float> ops::IndexMut<QKV> for Weight<T> {
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        use QKV::*;
        match index {
            Key => &mut self.key,
            Query => &mut self.query,
            Value => &mut self.value,
        }
    }
}

impl<T: Float + 'static> ops::Mul<Weight<T>> for Array2<T> {
    type Output = Weight<T>;

    fn mul(self, rhs: Weight<T>) -> Self::Output {
        let mut ctx = rhs.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = self.dot(&ctx[qkv]);
        }
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<Weight<T>> for &Array2<T> {
    type Output = Weight<T>;

    fn mul(self, rhs: Weight<T>) -> Self::Output {
        let mut ctx = rhs.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = self.dot(&ctx[qkv]);
        }
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<&Weight<T>> for &Array2<T> {
    type Output = Weight<T>;

    fn mul(self, rhs: &Weight<T>) -> Self::Output {
        let mut ctx = rhs.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = self.dot(&ctx[qkv]);
        }
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<Array2<T>> for Weight<T> {
    type Output = Self;

    fn mul(self, rhs: Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = ctx[qkv].dot(&rhs);
        }
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<&Array2<T>> for Weight<T> {
    type Output = Self;

    fn mul(self, rhs: &Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = ctx[qkv].dot(rhs);
        }
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<&Array2<T>> for &Weight<T> {
    type Output = Weight<T>;

    fn mul(self, rhs: &Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        for qkv in QKV::iter() {
            ctx[qkv] = ctx[qkv].dot(rhs);
        }
        ctx
    }
}

impl<T: Float + 'static> ops::MulAssign<Array2<T>> for Weight<T> {
    fn mul_assign(&mut self, rhs: Array2<T>) {
        for qkv in QKV::iter() {
            self[qkv] = self[qkv].dot(&rhs);
        }
    }
}

impl<T: Float + 'static> ops::MulAssign<&Array2<T>> for Weight<T> {
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        for qkv in QKV::iter() {
            self[qkv] = self[qkv].dot(rhs);
        }
    }
}

impl<T: Float + 'static> ops::MulAssign<&Array2<T>> for &mut Weight<T> {
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        for qkv in QKV::iter() {
            self[qkv] = self[qkv].dot(rhs);
        }
    }
}
