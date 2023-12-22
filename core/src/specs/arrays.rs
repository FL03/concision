/*
   Appellation: base <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Axis, Dimension, Ix2, NdFloat};
use ndarray::IntoDimension;
// use ndarray::linalg::Dot;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Bernoulli, BernoulliError, Uniform};
use ndarray_rand::RandomExt;
use num::Float;

pub trait Arange<T> {
    fn arange(start: T, stop: T, step: T) -> Self;
}

impl<T> Arange<T> for Vec<T>
where
    T: Float,
{
    fn arange(start: T, stop: T, step: T) -> Self {
        let n = ((stop - start) / step).ceil().to_usize().unwrap();
        (0..n).map(|i| start + step * T::from(i).unwrap()).collect()
    }
}

impl<T> Arange<T> for Array<T, Ix2>
where
    T: Float,
{
    fn arange(start: T, stop: T, step: T) -> Self {
        let n = ((stop - start) / step).ceil().to_usize().unwrap();
        let mut a = Array::zeros((n, 1));
        for (i, v) in a.iter_mut().enumerate() {
            *v = start + step * T::from(i).unwrap();
        }
        a
    }
}

pub trait GenerateRandom<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float + SampleUniform,
{
    fn bernoulli(
        dim: impl IntoDimension<Dim = D>,
        p: Option<f64>,
    ) -> Result<Array<bool, D>, BernoulliError> {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Array::random(dim.into_dimension(), dist))
    }

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = D>) -> Array<T, D> {
        let dim = dim.into_dimension();
        let dk = (T::one() / T::from(dim[axis]).unwrap()).sqrt();
        Self::uniform_between(dk, dim)
    }

    fn uniform_between(dk: T, dim: impl IntoDimension<Dim = D>) -> Array<T, D> {
        Array::random(dim, Uniform::new(-dk, dk))
    }
}

impl<T, D> GenerateRandom<T, D> for Array<T, D>
where
    T: Float + SampleUniform,
    D: Dimension,
{
}

pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}

pub trait Inverse<T = f64>: Sized
where
    T: Float,
{
    fn inverse(&self) -> Option<Self>;
}

impl<T> Inverse<T> for Array<T, Ix2>
where
    T: NdFloat,
{
    fn inverse(&self) -> Option<Self> {
        crate::compute_inverse(self)
    }
}
