/*
   Appellation: multi <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{attention::*, params::*, utils::*};

pub(crate) mod attention;
pub(crate) mod params;

use crate::attention::Weight;
use crate::core::prelude::BoxResult;
use crate::neural::prelude::Mask;
use crate::ops::Split;
use ndarray::prelude::Array2;
use ndarray::ScalarOperand;
use num::Float;

pub trait MultiHead<T>
where
    T: Float + ScalarOperand,
{
    fn attention(&mut self, data: &Array2<T>, mask: &Mask<T>) -> BoxResult<Array2<T>> {
        let weighted = data * self.weights();
        let (q, k, v) = weighted.split(self.params().heads())?;
        let score = utils::multihead(&q, &k, &v, mask)?;
        Ok(score)
    }

    fn params(&self) -> MultiHeadParams;

    fn weights(&self) -> &Weight<T>;
}

pub(crate) mod utils {
    use crate::attention::attention;
    use crate::neural::prelude::Mask;
    use crate::ops::Merge;
    use ndarray::prelude::{s, Array2, Array3, Array4};
    use ndarray::{ScalarOperand, ShapeError};
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
                let head = attention(&q, &k, &v, Some(mask.clone()));
                score.slice_mut(s![i, h, .., ..]).assign(&head);
            }
        }
        score.merge()
    }

    pub fn multihead<T>(
        query: &Array3<T>,
        key: &Array3<T>,
        value: &Array3<T>,
        mask: &Mask<T>,
    ) -> Result<Array2<T>, ShapeError>
    where
        T: Float + ScalarOperand,
    {
        let (heads, _, _) = query.dim();
        let mut score = Array3::<T>::zeros(query.dim());
        for h in 0..heads {
            let pos = s![h, .., ..];
            let q = query.slice(pos).to_owned();
            let k = key.slice(pos).to_owned();
            let v = value.slice(pos).to_owned();
            let head = attention(&q, &k, &v, mask.clone().into());
            score.slice_mut(s![h, .., ..]).assign(&head);
        }
        score.merge()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::prelude::Mask;

    #[test]
    fn test_multihead_shape() {
        let (heads, seq, model) = (8, 10, 512);
        let data = Array2::<f64>::zeros((seq, model));

        let mask = Mask::<f64>::masked(seq).into();
        let attention = MultiHeadAttention::new(heads, model);
        let score = attention
            .attention(&data, &mask)
            .expect("Failed to compute attention");
        assert_eq!(score.dim(), (seq, model));
    }
}
