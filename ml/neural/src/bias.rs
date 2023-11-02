/*
    Appellation: bias <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;
use ndarray_rand::rand_distr::{uniform::SampleUniform, Uniform};
use ndarray_rand::RandomExt;
use num::Float;

fn _generate_bias<T: Float + SampleUniform>(size: usize) -> Array1<T> {
    let ds = (T::from(size).unwrap()).sqrt();
    let dist = Uniform::new(-ds, ds);
    Array1::<T>::random(size, dist)
}

pub enum Bias<T: Float = f64> {
    Biased(Array1<T>),
    Unbiased,
}

impl<T: Float> Bias<T> {
    pub fn biased(size: usize) -> Self
    where
        T: SampleUniform,
    {
        let bias = _generate_bias(size);
        Self::Biased(bias)
    }

    pub fn forward(&self, data: &Array1<T>) -> Array1<T> {
        match self {
            Self::Biased(bias) => data + bias,
            Self::Unbiased => data.clone(),
        }
    }
}
