/*
   Appellation: split <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array2, Array3, Array4};
use ndarray::{Dimension, ShapeError};

// pub fn split<D: Dimension, T: Clone>(param: &Array<T, D>, heads: usize) -> Result<Array3<T>, ShapeError> {
//     let mut dim = param.dim()
//     let query = param.shape().last().unwrap() / heads;
//     // reshape the qkv matrix into a 3d array
//     let mut res = param.clone().into_shape((param.shape()[0], heads, query))?;
//     // swap the sequence and head axes
//     res.swap_axes(0, 1);
//     Ok(res)
// }

pub trait Split<T> {
    type Error;

    fn split(&self, heads: usize) -> Result<T, Self::Error>;
}

impl<T: Clone> Split<Array3<T>> for Array2<T> {
    type Error = ShapeError;

    fn split(&self, heads: usize) -> Result<Array3<T>, Self::Error> {
        let (seq, model) = self.dim();
        let query = model / heads;
        // reshape the qkv matrix into a 3d array
        let mut res = self.clone().into_shape((seq, heads, query))?;
        // swap the sequence and head axes
        res.swap_axes(0, 1);
        Ok(res)
    }
}

impl<T: Clone> Split<Array4<T>> for Array3<T> {
    type Error = ShapeError;

    fn split(&self, heads: usize) -> Result<Array4<T>, Self::Error> {
        let (batch, seq, model) = self.dim();
        let query = model / heads;
        // reshape the qkv matrix into a 3d array
        let mut res = self.clone().into_shape((batch, seq, heads, query))?;
        // swap the sequence and head axes
        res.swap_axes(1, 2);
        Ok(res)
    }
}
