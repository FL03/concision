/*
   Appellation: multi <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{attention::*, utils::*};

pub(crate) mod attention;

use crate::attention::params::MultiShape;
use crate::attention::AttentionHead;
// use crate::core::concat_iter;
use ndarray::prelude::{Array2, Array4};
use ndarray::{concatenate, Axis};

pub trait MultiHead {
    fn dim(&self) -> MultiShape;

    fn attention(&self) -> &[AttentionHead];

    fn heads_mut(&mut self) -> &mut [AttentionHead];

    fn process(&mut self, data: &Array2<f64>);

    fn score(&mut self, data: &Array2<f64>) -> Array2<f64> {
        let scores = self
            .attention()
            .iter()
            .map(|head| head.score())
            .collect::<Vec<_>>();
        let mut score: Array2<f64> = scores[0].clone();
        for i in 1..scores.len() {
            score = concatenate!(Axis(0), score, scores[i].clone());
        }
        score
    }
}

pub(crate) mod utils {
    use crate::attention::compute_attention;
    use crate::attention::ops::Merge;
    use ndarray::prelude::{Array2, Array3, Array4};
    use ndarray::{s, ShapeError};

    pub fn multihead(
        query: &Array4<f64>,
        key: &Array4<f64>,
        value: &Array4<f64>,
        mask: Option<Array2<f64>>,
    ) -> Result<Array3<f64>, ShapeError> {
        let (batch, heads, seq, _) = query.dim();
        let mask = mask.unwrap_or_else(|| Array2::<f64>::zeros((seq, seq)));
        let mut score = Array4::<f64>::zeros(query.dim());
        for i in 0..batch {
            for h in 0..heads {
                let q = query.slice(s![i, h, .., ..]).to_owned();
                let k = key.slice(s![i, h, .., ..]).to_owned();
                let v = value.slice(s![i, h, .., ..]).to_owned();
                let head = compute_attention(&q, &k, &v, Some(mask.clone()));
                score.slice_mut(s![i, h, .., ..]).assign(&head);
            }
        }
        score.merge()
    }
}

#[cfg(test)]
mod tests {}
