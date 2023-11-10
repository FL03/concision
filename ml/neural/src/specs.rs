/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Array2};
use ndarray::{Dimension, IntoDimension};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Bernoulli, BernoulliError, Uniform};
use ndarray_rand::RandomExt;
use num::Float;

pub trait InitRandom<T = f64>
where
    T: Float + SampleUniform,
{
    type Dim: Dimension;

    fn bernoulli(
        dim: impl IntoDimension<Dim = Self::Dim>,
        p: Option<f64>,
    ) -> Result<Array<bool, Self::Dim>, BernoulliError> {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Array::random(dim.into_dimension(), dist))
    }

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = Self::Dim>) -> Array<T, Self::Dim> {
        let dim = dim.into_dimension();
        let k = (T::one() / T::from(dim[axis]).unwrap()).sqrt();
        Array::random(dim, Uniform::new(-k, k))
    }
}

impl<T, D> InitRandom<T> for Array<T, D>
where
    T: Float + SampleUniform,
    D: Dimension,
{
    type Dim = D;

    fn bernoulli(
        dim: impl IntoDimension<Dim = Self::Dim>,
        p: Option<f64>,
    ) -> Result<Array<bool, Self::Dim>, BernoulliError> {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Array::random(dim.into_dimension(), dist))
    }

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = Self::Dim>) -> Array<T, Self::Dim> {
        let dim = dim.into_dimension();
        let k = (T::from(dim[axis]).unwrap()).sqrt();
        Array::random(dim, Uniform::new(-k, k))
    }
}

pub trait Trainable<T: Float> {
    fn train(&mut self, args: &Array2<T>) -> Array2<T>;
}
