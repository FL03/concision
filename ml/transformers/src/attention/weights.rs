/*
   Appellation: weights <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::Dim;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

pub type WeightsArray = ndarray::Array2<f64>;

pub type WeightDim = Dim<[usize; 2]>;

fn compute_head_size(depth: usize, heads: usize) -> usize {
    depth / heads
}

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
pub enum Weights {
    #[serde(alias = "k")]
    Key,
    #[default]
    #[serde(alias = "q")]
    Query,
    #[serde(alias = "v")]
    Value,
}

impl ndarray::Dimension for Weights {
    const NDIM: usize = 3;

    fn ndim(&self) -> usize {
        3
    }
    fn shape(&self) -> ndarray::IxDyn {
        ndarray::IxDyn(&[1])
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct Weight {
    key: Vec<f64>,
    query: Vec<f64>,
    value: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_weights() {
        let w = Weights::Key;
        assert_eq!(w.to_string(), "key");
        assert_eq!(Weights::Key, Weights::from_str("key").unwrap());
        assert_eq!(Weights::Key, Weights::from_str("k").unwrap());
    }
}