/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array;
use ndarray::{Dimension, IntoDimension};
// use ndarray::linalg::Dot;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Bernoulli, BernoulliError, Uniform};
use ndarray_rand::RandomExt;
use num::{Float, One, Zero};

pub trait Borrowed<T>: AsRef<T> + AsMut<T> {}

impl<S, T> Borrowed<T> for S where S: AsRef<T> + AsMut<T> {}

pub trait BinaryNum: One + Zero {}

impl<T> BinaryNum for T where T: One + Zero {}

pub trait Pair<A, B> {
    fn pair(&self) -> (A, B);
}

impl<A, B, T> Pair<A, B> for T
where
    T: Clone + Into<(A, B)>,
{
    fn pair(&self) -> (A, B) {
        self.clone().into()
    }
}

pub trait GenerateRandom<T = f64>
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
        let dk = (T::one() / T::from(dim[axis]).unwrap()).sqrt();
        Self::uniform_between(dk, dim)
    }

    fn uniform_between(dk: T, dim: impl IntoDimension<Dim = Self::Dim>) -> Array<T, Self::Dim> {
        Array::random(dim, Uniform::new(-dk, dk))
    }
}

impl<T, D> GenerateRandom<T> for Array<T, D>
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
