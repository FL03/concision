/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Axis, Dimension, Ix2, NdFloat};
use ndarray::IntoDimension;
// use ndarray::linalg::Dot;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Bernoulli, BernoulliError, Uniform};
use ndarray_rand::RandomExt;
use num::{Float, FromPrimitive, One, Signed, Zero};
use std::ops;

pub trait Arithmetic<S, T>:
    ops::Add<S, Output = T>
    + ops::Div<S, Output = T>
    + ops::Mul<S, Output = T>
    + ops::Sub<S, Output = T>
{
}

impl<A, S, T> Arithmetic<S, T> for A where
    A: ops::Add<S, Output = T>
        + ops::Div<S, Output = T>
        + ops::Mul<S, Output = T>
        + ops::Sub<S, Output = T>
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

pub trait Apply<T> {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T;
}

impl<T, D> Apply<T> for Array<T, D>
where
    D: Dimension,
{
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        self.map(f)
    }
}

pub trait ApplyTo<T> {
    fn apply_to(&self, args: &mut T) -> &mut T;
}

pub trait As<T>: AsRef<T> + AsMut<T> {}

impl<S, T> As<T> for S where S: AsRef<T> + AsMut<T> {}

pub trait BinaryNum: One + Zero {}

impl<T> BinaryNum for T where T: One + Zero {}

pub trait FloatExt: FromPrimitive + NdFloat + Signed + SampleUniform {}

impl<T> FloatExt for T where T: FromPrimitive + NdFloat + Signed + SampleUniform {}

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

pub trait MatrixOps<T = f64, A = Ix2, B = Ix2>:
    Arithmetic<Array<T, A>, Array<T, B>> + Sized
where
    A: Dimension,
    B: Dimension,
{
}

impl<T, D, A, B> MatrixOps<T, A, B> for Array<T, D>
where
    A: Dimension,
    B: Dimension,
    D: Dimension,
    T: Float,
    Self: Arithmetic<Array<T, A>, Array<T, B>>,
{
}

impl<T, D, A, B> MatrixOps<T, A, B> for &Array<T, D>
where
    A: Dimension,
    B: Dimension,
    D: Dimension,
    T: Float,
    Self: Arithmetic<Array<T, A>, Array<T, B>>,
{
}
