/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use crate::layers::Features;
use ndarray::prelude::{Array, Array1, Array2, Dimension, Ix2};
use ndarray::{IntoDimension, RemoveAxis};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

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
