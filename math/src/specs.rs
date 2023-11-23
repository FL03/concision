/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array;
use ndarray::{Dimension, ShapeError};
use num::Num;
use std::ops;

pub trait NumOpsAssign:
    Sized + ops::AddAssign + ops::DivAssign + ops::MulAssign + ops::SubAssign
{
}

impl<T> NumOpsAssign for T where T: ops::AddAssign + ops::DivAssign + ops::MulAssign + ops::SubAssign
{}

trait Matmul<T, D>
where
    T: Num,
    D: Dimension,
{
    fn matmul(&self, other: &Array<T, D>) -> Result<Array<T, D>, ShapeError>;

    fn shape(&self) -> D;
}

// impl<T, D> Matmul<T, D> for Array<T, D>
// where
//     T: Num + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Clone,
//     D: Dimension,
// {
//     fn matmul(&self, other: &Array<T, D>) -> Result<Array<T, D>, ShapeError> {
//         let self_shape = self.shape();
//         let other_shape = other.shape();

//         if self_shape[self.ndim() - 1] != other_shape[self.ndim() - 2] {
//             return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape));
//         }

//         let mut result = Array::zeros(self_shape);

//         let mut self_shape = self_shape.to_vec();
//         let self_last = self_shape.pop().unwrap();
//         let other_shape = other_shape.to_vec();

//         let mut iter_self = self.iter();
//         let mut iter_other = other.iter();

//         for mut row_result in result.outer_iter_mut() {
//             for mut col_other in other.inner_iter() {
//                 let row_self = iter_self.clone();
//                 let mut col_other = col_other.clone();
//                 let dot = dot_product(&mut row_self, &mut col_other, self_last, &other_shape);
//                 row_result.assign(&dot);
//             }
//             iter_self.step_by(self_shape.last().unwrap().index());
//         }

//         Ok(result)
//     }

//     fn shape(&self) -> D {
//         self.raw_dim()
//     }
// }

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_product() {
        let args = vec![2.0, 4.0, 6.0];
        assert_eq!(args.into_iter().product::<f64>(), 48.0);
    }
}
