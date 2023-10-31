/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::Float;

pub trait Bias<T: Float + SampleUniform> {
    fn init_uniform(features: usize) -> Array1<T> {
        let k = (T::from(features).unwrap()).sqrt();
        let uniform = Uniform::new(-k, k);
        Array1::random(features, uniform)
    }

    fn bias(&self) -> &Array1<T>;
    fn bias_mut(&mut self) -> &mut Array1<T>;
}

pub trait Trainable {
    fn train(&mut self, args: &[f64]) -> f64;
}
