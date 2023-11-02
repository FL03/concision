/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::utils::multihead;
use super::MultiHeadParams;
use crate::attention::Weight;
use crate::neural::prelude::Forward;
use crate::ops::Split;
use ndarray::prelude::Array2;

pub struct MultiHeadAttention {
    mask: Array2<f64>,
    params: MultiHeadParams,
    weights: Weight,
}

impl MultiHeadAttention {
    pub fn new(heads: usize, model: usize) -> Self {
        let params = MultiHeadParams::new(heads, model);
        let mask = Array2::<f64>::zeros((params.model, params.model));
        let weights = Weight::new((params.model, params.model));
        Self {
            mask,
            params,
            weights,
        }
    }

    pub fn attention(&self, data: &Array2<f64>) -> Array2<f64> {
        let weighted = self.weights() * data;
        let (q, k, v) = weighted.split(self.params().heads()).unwrap();
        let score = multihead(&q, &k, &v, Some(self.mask().clone())).unwrap();
        score
    }

    pub fn mask(&self) -> &Array2<f64> {
        &self.mask
    }

    pub fn params(&self) -> MultiHeadParams {
        self.params
    }

    pub fn weights(&self) -> &Weight {
        &self.weights
    }
}

impl Forward<Array2<f64>> for MultiHeadAttention {
    type Output = Array2<f64>;

    fn forward(&self, data: &Array2<f64>) -> Self::Output {
        self.attention(data)
    }
}
