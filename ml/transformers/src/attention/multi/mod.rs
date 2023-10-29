/*
   Appellation: multi <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{attention::*, utils::*};

pub(crate) mod attention;

use crate::ops::Split;
use crate::attention::params::MultiShape;
use crate::attention::Weight;
use crate::core::prelude::BoxResult;
use ndarray::prelude::{Array2, Array3};
use ndarray::ScalarOperand;
use num::Float;

pub trait MultiHead<T>
where
    T: Float + ScalarOperand,
{
    fn attention(&mut self, data: &Array2<T>) -> BoxResult<Array2<T>> {
        let weighted = self.weights() * data;
        let (q, k, v) = weighted.split(self.dim().heads())?;
        let score = utils::multihead(&q, &k, &v, Some(self.mask().clone()))?;
        Ok(score)
    }
    
    fn dim(&self) -> MultiShape;

    fn mask(&self) -> &Array2<T>;

    fn multihead(&self) -> &Array3<T>;

    fn multihead_mut(&mut self) -> &mut Array3<T>;

    fn weights(&self) -> &Weight<T>;

    
}

pub(crate) mod utils {
    use crate::attention::compute_attention;
    use crate::ops::Merge;
    use ndarray::prelude::{Array2, Array3, Array4};
    use ndarray::{s, ScalarOperand, ShapeError};
    use num::Float;

    pub fn batched_multihead(
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

    pub fn multihead<T>(
        query: &Array3<T>,
        key: &Array3<T>,
        value: &Array3<T>,
        mask: Option<Array2<T>>,
    ) -> Result<Array2<T>, ShapeError>
    where
        T: Float + ScalarOperand,
    {
        let (heads, seq, _) = query.dim();
        let mask = mask.unwrap_or_else(|| Array2::<T>::zeros((seq, seq)));
        let mut score = Array3::<T>::zeros(query.dim());
        for h in 0..heads {
            let pos = s![h, .., ..];
            let q = query.slice(pos).to_owned();
            let k = key.slice(pos).to_owned();
            let v = value.slice(pos).to_owned();
            let head = compute_attention(&q, &k, &v, Some(mask.clone()));
            score.slice_mut(s![h, .., ..]).assign(&head);
        }
        score.merge()
    }
}

#[cfg(test)]
mod tests {}
