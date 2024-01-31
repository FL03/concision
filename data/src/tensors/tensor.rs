/*
   Appellation: tensor <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::TensorKind;
use crate::core::id::AtomicId;
use crate::prelude::DType;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Num;
use serde::{Deserialize, Serialize};
use std::ops;

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct Tensor<T = f64, D = Ix2>
where
    D: Dimension,
{
    id: AtomicId,
    data: Array<T, D>,
    dtype: DType,
    kind: TensorKind,
}

impl<T, D> Tensor<T, D>
where
    D: Dimension,
{
    pub fn new(data: Array<T, D>, kind: TensorKind) -> Self {
        Self {
            id: AtomicId::new(),
            data,
            dtype: DType::default(),
            kind,
        }
    }

    pub fn mode(&self) -> TensorKind {
        self.kind
    }

    pub fn set_mode(&mut self, mode: TensorKind) {
        self.kind = mode;
    }

    pub fn as_variable(mut self) -> Self {
        self.kind = TensorKind::Variable;
        self
    }
}

impl<T, D> Tensor<T, D>
where
    D: Dimension,
    T: Clone + Default + Num + Into<DType>,
{
    pub fn ones(shape: impl IntoDimension<Dim = D>) -> Self {
        Self {
            id: AtomicId::new(),
            data: Array::ones(shape),
            dtype: T::default().into(),
            kind: TensorKind::default(),
        }
    }

    pub fn zeros(shape: impl IntoDimension<Dim = D>) -> Self {
        Self {
            id: AtomicId::new(),
            data: Array::zeros(shape),
            dtype: T::default().into(),
            kind: TensorKind::default(),
        }
    }
}

impl<T, D> std::fmt::Display for Tensor<T, D>
where
    D: Dimension,
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl<T, D> ops::Deref for Tensor<T, D>
where
    D: Dimension,
{
    type Target = Array<T, D>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, D> ops::DerefMut for Tensor<T, D>
where
    D: Dimension,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
