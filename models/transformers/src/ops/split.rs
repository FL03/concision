/*
   Appellation: split <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, ArrayBase, Data, Dimension, RemoveAxis, ShapeError};

// pub fn split<D: Dimension, T: Clone>(param: &Array<T, D>, heads: usize) -> Result<Array3<T>, ShapeError> {
//     let mut dim = param.dim()
//     let query = param.shape().last().unwrap() / heads;
//     // reshape the qkv matrix into a 3d array
//     let mut res = param.clone().into_shape((param.shape()[0], heads, query))?;
//     // swap the sequence and head axes
//     res.swap_axes(0, 1);
//     Ok(res)
// }

pub trait DimSplit {
    type Output;

    fn split(&self, h: usize) -> Self::Output;
}

pub trait Split {
    type Output;

    fn split(&self, heads: usize) -> Result<Self::Output, ShapeError>;
}

/*
 ************* Implementations *************
*/

impl<D, E> DimSplit for D
where
    D: Dimension<Larger = E>,
    E: RemoveAxis<Smaller = D>,
{
    type Output = E;

    fn split(&self, h: usize) -> Self::Output {
        super::utils::_split_dim(self, h)
    }
}

impl<A, S, D, E> Split for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension<Larger = E>,
    E: RemoveAxis<Smaller = D>,
    S: Data<Elem = A>,
    ArrayBase<S, D>: Clone,
{
    type Output = Array<A, E>;

    fn split(&self, h: usize) -> Result<Self::Output, ShapeError> {
        super::_split(self, h, false)
    }
}

// impl<T: Clone> Split for Array2<T> {
//     type Output = Array3<T>;

//     fn split(&self, heads: usize) -> Result<Self::Output, ShapeError> {
//         let (seq, model) = self.dim();
//         let query = model / heads;
//         // reshape the qkv matrix into a 3d array
//         let mut res = self.clone().into_shape((seq, heads, query))?;
//         // swap the sequence and head axes
//         res.swap_axes(0, 1);
//         Ok(res)
//     }
// }

// impl<T: Clone> Split for Array3<T> {
//     type Output = Array4<T>;

//     fn split(&self, heads: usize) -> Result<Self::Output, ShapeError> {
//         let (batch, seq, model) = self.dim();
//         let query = model / heads;
//         // reshape the qkv matrix into a 3d array
//         let mut res = self.clone().into_shape((batch, seq, heads, query))?;
//         // swap the sequence and head axes
//         res.swap_axes(1, 2);
//         Ok(res)
//     }
// }
