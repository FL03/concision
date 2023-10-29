/*
   Appellation: split <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array2, Array3, Array4};
use ndarray::ShapeError;

pub trait Merge<T> {
    type Error;

    fn merge(&self) -> Result<T, Self::Error>;
}

impl<T: Clone> Merge<Array2<T>> for Array3<T> {
    type Error = ShapeError;

    fn merge(&self) -> Result<Array2<T>, Self::Error> {
        let (heads, seq, query) = self.dim();
        let mut tmp = self.clone();
        // swap the head and sequence axes
        tmp.swap_axes(0, 1);
        // reshape the qkv matrix into a 2d array
        tmp.into_shape((seq, heads * query))
    }
}

impl<T: Clone> Merge<Array3<T>> for Array4<T> {
    type Error = ShapeError;

    fn merge(&self) -> Result<Array3<T>, Self::Error> {
        let (batch, heads, seq, query) = self.dim();
        let mut tmp = self.clone();
        // swap the head and sequence axes
        tmp.swap_axes(1, 2);
        // reshape the qkv matrix into a 2d array
        tmp.into_shape((batch, seq, heads * query))
    }
}
