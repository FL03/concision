/*
   Appellation: embedding <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array2, Dim, IntoDimension};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct Embedding {
    data: Array2<f64>,
}

impl Embedding {
    pub fn new(dim: Dim<[usize; 2]>) -> Self {
        Self {
            data: Array2::zeros(dim),
        }
    }
}

impl std::fmt::Display for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<D> From<D> for Embedding
where
    D: IntoDimension<Dim = Dim<[usize; 2]>>,
{
    fn from(dim: D) -> Self {
        Self::new(dim.into_dimension())
    }
}
