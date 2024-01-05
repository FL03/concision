/*
   Appellation: base <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Axis, Dimension, Ix1, Ix2, NdFloat};
use ndarray::{IntoDimension, ScalarOperand, ShapeError};
// use ndarray::linalg::Dot;
use distr::uniform::SampleUniform;
use distr::{Bernoulli, BernoulliError, Distribution, StandardNormal, Uniform};
use ndarray_rand::rand_distr as distr;
use ndarray_rand::RandomExt;

use num::{Float, Num};
use std::ops;

pub trait Affine<T = f64>: Sized {
    type Error;

    fn affine(&self, mul: T, add: T) -> Result<Self, Self::Error>;
}

impl<S, T, D> Affine<S> for Array<T, D>
where
    T: Num + ScalarOperand,
    D: Dimension,
    Array<T, D>: ops::Mul<S, Output = Array<T, D>> + ops::Add<S, Output = Array<T, D>>,
{
    type Error = ShapeError;

    fn affine(&self, mul: S, add: S) -> Result<Self, Self::Error> {
        Ok(self.clone() * mul + add)
    }
}

pub enum ArangeArgs<T> {
    Arange { start: T, stop: T, step: T },
    Between { start: T, stop: T },
    Until { stop: T },
}

impl<T> From<(T, T, T)> for ArangeArgs<T> {
    fn from(args: (T, T, T)) -> Self {
        ArangeArgs::Arange {
            start: args.0,
            stop: args.1,
            step: args.2,
        }
    }
}

impl<T> From<(T, T)> for ArangeArgs<T> {
    fn from(args: (T, T)) -> Self {
        ArangeArgs::Between {
            start: args.0,
            stop: args.1,
        }
    }
}

impl<T> From<T> for ArangeArgs<T>
where
    T: Num,
{
    fn from(args: T) -> Self {
        ArangeArgs::Until { stop: args }
    }
}

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

impl<T> Arange<T> for Array<T, Ix1>
where
    T: Float,
{
    fn arange(start: T, stop: T, step: T) -> Self {
        let n = ((stop - start) / step).ceil().to_usize().unwrap();
        Array::from_shape_fn(n, |i| start + step * T::from(i).unwrap())
    }
}

impl<T> Arange<T> for Array<T, Ix2>
where
    T: Float,
{
    fn arange(start: T, stop: T, step: T) -> Self {
        let n = ((stop - start) / step).ceil().to_usize().unwrap();
        Array::from_shape_fn((n, 1), |(i, ..)| start + step * T::from(i).unwrap())
    }
}

pub trait GenerateRandom<T = f64, D = Ix2>: Sized
where
    D: Dimension,
    T: Float,
{
    fn rand<IdS>(dim: impl IntoDimension<Dim = D>, distr: IdS) -> Self
    where
        IdS: Distribution<T>;

    fn bernoulli(dim: impl IntoDimension<Dim = D>, p: Option<f64>) -> Result<Self, BernoulliError>
    where
        Bernoulli: Distribution<T>,
    {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Self::rand(dim.into_dimension(), dist))
    }

    fn stdnorm(dim: impl IntoDimension<Dim = D>) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        Self::rand(dim, StandardNormal)
    }

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: SampleUniform,
    {
        let dim = dim.into_dimension();
        let dk = T::from(dim[axis]).unwrap().recip().sqrt();
        Self::uniform_between(dk, dim)
    }

    fn uniform_between(dk: T, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: SampleUniform,
    {
        Self::rand(dim, Uniform::new(-dk, dk))
    }
}

pub trait GenerateExt<T = f64, D = Ix2>: GenerateRandom<T, D>
where
    D: Dimension,
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    type Output;
    // fn bernoulli(
    //     dim: impl IntoDimension<Dim = D>,
    //     p: Option<f64>,
    // ) -> Result<Self::Output, BernoulliError> {
    //     let dist = Bernoulli::new(p.unwrap_or(0.5))?;
    //     Ok(Array::random(dim.into_dimension(), dist))
    // }
}

impl<T, D> GenerateRandom<T, D> for Array<T, D>
where
    T: Float + SampleUniform,
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    fn rand<IdS>(dim: impl IntoDimension<Dim = D>, distr: IdS) -> Self
    where
        IdS: Distribution<T>,
    {
        Self::random(dim.into_dimension(), distr)
    }
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

// pub trait Stack

pub trait Genspace<T = f64> {
    fn arange(start: T, stop: T, step: T) -> Self;

    fn linspace(start: T, stop: T, n: usize) -> Self;

    fn logspace(start: T, stop: T, n: usize) -> Self;

    fn geomspace(start: T, stop: T, n: usize) -> Self;

    fn ones(n: usize) -> Self;

    fn zeros(n: usize) -> Self;
}

pub trait ArrayLike {
    fn ones_like(&self) -> Self;

    fn zeros_like(&self) -> Self;
}

impl<T, D> ArrayLike for Array<T, D>
where
    T: Clone + Num,
    D: Dimension,
{
    fn ones_like(&self) -> Self {
        Array::ones(self.dim())
    }

    fn zeros_like(&self) -> Self {
        Array::zeros(self.dim())
    }
}
