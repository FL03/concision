/*
    Appellation: param <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Param, ParamKind};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Float;
use uuid::Uuid;

#[cfg(feature = "rand")]
pub(crate) fn gen_id() -> Uuid {
    Uuid::new_v4()
}

#[cfg(not(feature = "rand"))]
pub(crate) fn gen_id() -> Uuid {
    uuid::Uuid::new_v8()
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Parameter<T = f64, D = Ix2>
where
    D: Dimension,
{
    pub(crate) id: String,
    pub(crate) features: D,
    pub(crate) kind: ParamKind,
    pub(crate) name: String,
    pub(crate) value: Array<T, D>,
}

impl<T, D> Parameter<T, D>
where
    D: Dimension,
{
    pub fn new(features: impl IntoDimension<Dim = D>, kind: ParamKind, name: impl ToString) -> Self
    where
        T: Clone + Default,
    {
        let features = features.into_dimension();
        Self {
            id: gen_id().to_string(),
            features: features.clone(),
            kind,
            name: name.to_string(),
            value: Array::default(features),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
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

    pub fn data(&self) -> &Array<T, D> {
        &self.value
    }

    pub fn params_mut(&mut self) -> &mut Array<T, D> {
        &mut self.value
    }

    pub fn set_kind(&mut self, kind: ParamKind) {
        self.kind = kind;
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn set_params(&mut self, params: Array<T, D>) {
        self.value = params;
    }

    pub fn with_kind(self, kind: ParamKind) -> Self {
        Self { kind, ..self }
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            ..self
        }
    }

    pub fn with_params(self, params: Array<T, D>) -> Self {
        Self {
            value: params,
            ..self
        }
    }
}

impl<T, D> Param for Parameter<T, D>
where
    T: Float,
    D: Dimension,
{
    type Key = ParamKind;
    type Value = Array<T, D>;
}

impl<S, T, D, O> Dot<S> for Parameter<T, D>
where
    Array<T, D>: Dot<S, Output = O>,
    D: Dimension,
    T: Float,
{
    type Output = O;

    fn dot(&self, rhs: &S) -> Self::Output {
        self.value.dot(rhs)
    }
}
