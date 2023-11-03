/*
   Appellation: merge <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array2, Array3, Array4, Axis};
use ndarray::ShapeError;

pub trait Merge<T> {
    type Error;

    fn merge(&self) -> Result<T, Self::Error>;
}

impl<T> Merge<Array2<T>> for Array3<T> where T: Clone {
    type Error = ShapeError;

    fn merge(&self) -> Result<Array2<T>, Self::Error> {
        if self.ndim() < 3 {
            return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape));
        }
        let axes = (self.ndim() - 3, self.ndim() - 2, self.ndim() - 1);

        let mut tmp = self.clone();
        // swap the head and sequence axes
        tmp.swap_axes(axes.0, axes.1);
        // reshape the qkv matrix into a 2d array
        if tmp.merge_axes(Axis(axes.1), Axis(axes.2)) {
            let res = tmp.remove_axis(Axis(axes.1));
            Ok(res)
        } else {
            Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape))
        }
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
