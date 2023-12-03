/*
   Appellation: transformer <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::Transform;
use ndarray::prelude::{Array2, NdFloat};

#[derive(Clone, Debug, Default)]
pub struct Transformer;

impl<T> Transform<Array2<T>> for Transformer
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn transform(&self, args: &Array2<T>) -> Self::Output {
        args.clone()
    }
}
