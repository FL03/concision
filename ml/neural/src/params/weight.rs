/*
   Appellation: weight <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array2;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Weight<T = f64> {
    weights: Array2<T>,
}

impl<T> Weight<T>
where
    T: Default,
{
    pub fn new(m: usize, n: usize) -> Self {
        let weights = Array2::default((m, n));
        Self { weights }
    }
}

impl<T> Weight<T>
where
    T: Float + SampleUniform,
{
    pub fn uniform(m: usize, n: usize) -> Array2<T> {
        let dk = (T::from(m).unwrap()).sqrt();
        let dist = Uniform::new(-dk, dk);
        Array2::random((m, n), dist)
    }
}
