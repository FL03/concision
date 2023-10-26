/*
   Appellation: context <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array2, IntoDimension, Ix2};
use serde::{Deserialize, Serialize};
use std::ops;
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumIs,
    EnumIter,
    EnumString,
    EnumVariantNames,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum QKV {
    #[serde(alias = "k")]
    Key,
    #[default]
    #[serde(alias = "q")]
    Query,
    #[serde(alias = "v")]
    Value,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Context {
    dim: Ix2,
    pub key: Array2<f64>,
    pub query: Array2<f64>,
    pub value: Array2<f64>,
}

impl Context {
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

    pub fn with_weight(&mut self, query: &Array2<f64>, key: &Array2<f64>, value: &Array2<f64>) {
        self.key = key.clone();
        self.query = query.clone();
        self.value = value.clone();
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<D> From<D> for Context
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

impl From<Context> for (Array2<f64>, Array2<f64>, Array2<f64>) {
    fn from(context: Context) -> Self {
        (context.key, context.query, context.value)
    }
}

impl ops::Index<QKV> for Context {
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

impl ops::IndexMut<QKV> for Context {
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        match index {
            QKV::Key => &mut self.key,
            QKV::Query => &mut self.query,
            QKV::Value => &mut self.value,
        }
    }
}

impl ops::Mul<&Array2<f64>> for Context {
    type Output = Self;

    fn mul(self, rhs: &Array2<f64>) -> Self::Output {
        let mut context = self.clone();
        context.key = context.key.dot(rhs);
        context.query = context.query.dot(rhs);
        context.value = context.value.dot(rhs);
        context
    }
}

impl ops::Mul<&Array2<f64>> for &Context {
    type Output = Context;

    fn mul(self, rhs: &Array2<f64>) -> Self::Output {
        let mut context = self.clone();
        context.key = context.key.dot(rhs);
        context.query = context.query.dot(rhs);
        context.value = context.value.dot(rhs);
        context
    }
}

impl ops::MulAssign<Array2<f64>> for Context {
    fn mul_assign(&mut self, rhs: Array2<f64>) {
        self.key = self.key.dot(&rhs);
        self.query = self.query.dot(&rhs);
        self.value = self.value.dot(&rhs);
    }
}

impl ops::MulAssign<&Array2<f64>> for Context {
    fn mul_assign(&mut self, rhs: &Array2<f64>) {
        self.key = self.key.dot(rhs);
        self.query = self.query.dot(rhs);
        self.value = self.value.dot(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_attention_weights() {
        let w = QKV::Key;
        assert_eq!(w.to_string(), "key");
        assert_eq!(QKV::Key, QKV::from_str("key").unwrap());
        assert_eq!(QKV::Key, QKV::from_str("k").unwrap());
    }
}
