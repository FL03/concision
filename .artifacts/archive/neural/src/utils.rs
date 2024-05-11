/*
    Appellation: utils <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array;
use ndarray::{Dimension, IntoDimension};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::Float;

pub fn generate_uniform_arr<T, D>(axis: usize, dim: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: Float + SampleUniform,
{
    let shape: D = dim.into_dimension();
    let dk = {
        let k = T::from(shape[axis]).unwrap();
        (T::one() / k).sqrt()
    };
    let dist = Uniform::new(-dk, dk);
    Array::<T, D>::random(shape, dist)
}
