/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2};
use num::Float;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamGroup<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
    <D as Dimension>::Smaller: Dimension,
{
    bias: Array<T, D::Smaller>,
    weights: Array<T, D>,
}

// impl<T, D> ParamGroup<T, D>
// where
//     T: Float,
//     D: Dimension + RemoveAxis,
//     <D as Dimension>::Smaller: Dimension,
// {
//     pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
//         let dim = dim.into_dimension();
//         let bias = Array::zeros();
//         let weights = Array::zeros(dim.clone());
//         Self {
//             bias: Array::<T, D::Smaller>::zeros(&weights.shape()[..(dim.ndim() - 1)]),
//             weights,
//         }
//     }
// }
