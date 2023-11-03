/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{multihead, MultiHeadParams};
use crate::attention::Weight;
use crate::neural::prelude::{ Mask};
use crate::ops::Split;
use ndarray::{ScalarOperand, ShapeError};
use ndarray::prelude::Array2;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MultiHeadAttention<T: Float = f64> {
    params: MultiHeadParams,
    weights: Weight<T>,
}

impl<T: Float> MultiHeadAttention<T> {
    pub fn new(heads: usize, model: usize) -> Self {
        let params = MultiHeadParams::new(heads, model);
        let weights = Weight::new((model, model));
        Self {
            params,
            weights,
        }
    }

    pub fn params(&self) -> MultiHeadParams {
        self.params
    }

    pub fn weights(&self) -> &Weight<T> {
        &self.weights
    }
}

impl<T: Float + ScalarOperand> MultiHeadAttention<T> {
    pub fn attention(&self, data: &Array2<T>, mask: &Mask<T>) -> Result<Array2<T>, ShapeError> {
        let weighted = data * self.weights();
        let (q, k, v) = weighted.split(self.params().heads())?;
        let score = multihead(&q, &k, &v, mask)?;
        Ok(score)
    }
}

// impl<T: Float + 'static> Attention<T> for MultiHeadAttention<T> {
//     fn key(&self) -> &Array2<T> {
//         self.weights.key()
//     }

//     fn mask(&self) -> &Array2<T> {
//         &self.mask
//     }

//     fn query(&self) -> &Array2<T> {
//         &self.weights.query()
//     }

//     fn value(&self) -> &Array2<T> {
//         &self.weights.value()
//     }
// }

// impl<T: Float + ScalarOperand> Forward<Array2<T>> for MultiHeadAttention<T> {
//     type Output = Result<Array2<T>, ShapeError>;

//     fn forward(&self, data: &Array2<T>) -> Self::Output {
//         self.attention(data)
//     }
// }
