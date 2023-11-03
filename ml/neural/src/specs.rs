/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Array1, Array2};
use ndarray::{Dimension, IntoDimension};
use ndarray_rand::rand_distr as dist;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::RandomExt;
use num::Float;

pub trait InitUniform<T = f64>
where
    T: Float + SampleUniform,
{
    type Dim: Dimension;

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = Self::Dim>) -> Array<T, Self::Dim> {
        let dim = dim.into_dimension();
        let k = (T::from(dim[axis]).unwrap()).sqrt();
        let uniform = dist::Uniform::new(-k, k);
        Array::random(dim, uniform)
    }
}

pub trait Trainable<T: Float> {
    fn train(&mut self, args: &Array2<T>) -> Array2<T>;
}
