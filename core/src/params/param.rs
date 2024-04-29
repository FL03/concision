/*
    Appellation: param <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Param, ParamKind};
use crate::prelude::GenerateRandom;
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::Float;

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Parameter<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    features: D,
    kind: ParamKind,
    name: String,
    params: Array<T, D>,
}

impl<T, D> Parameter<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn new(
        features: impl IntoDimension<Dim = D>,
        kind: ParamKind,
        name: impl ToString,
    ) -> Self {
        let features = features.into_dimension();
        Self {
            features: features.clone(),
            kind,
            name: name.to_string(),
            params: Array::zeros(features),
        }
    }

    pub fn kind(&self) -> &ParamKind {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut ParamKind {
        &mut self.kind
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }

    pub fn params(&self) -> &Array<T, D> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Array<T, D> {
        &mut self.params
    }

    pub fn set_kind(&mut self, kind: ParamKind) {
        self.kind = kind;
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn set_params(&mut self, params: Array<T, D>) {
        self.params = params;
    }

    pub fn with_kind(mut self, kind: ParamKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn with_params(mut self, params: Array<T, D>) -> Self {
        self.params = params;
        self
    }
}

impl<T, D> Parameter<T, D>
where
    D: Dimension,
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init_uniform(mut self, dk: T) -> Self {
        let dim = self.params.dim();
        self.params = Array::uniform_between(dk, dim);
        self
    }
}

impl<T, D> Param for Parameter<T, D>
where
    T: Float,
    D: Dimension,
{
    type Dim = D;

    fn kind(&self) -> &ParamKind {
        &self.kind
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl<S, T, D, O> Dot<S> for Parameter<T, D>
where
    Array<T, D>: Dot<S, Output = O>,
    D: Dimension,
    T: Float,
{
    type Output = O;

    fn dot(&self, rhs: &S) -> Self::Output {
        self.params.dot(rhs)
    }
}
