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

use num::{Float, Num, ToPrimitive};
use num::traits::real::Real;
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

impl<T> ArangeArgs<T> where T: Copy + Num {
    /// Returns the start value of the range.
    pub fn start(&self) -> T {
        match self {
            ArangeArgs::Arange { start, .. } => *start,
            ArangeArgs::Between { start, .. } => *start,
            ArangeArgs::Until { .. } => T::zero(),
        }
    }
    /// Returns the stop value of the range.
    pub fn stop(&self) -> T {
        match self {
            ArangeArgs::Arange { stop, .. } => *stop,
            ArangeArgs::Between { stop, .. } => *stop,
            ArangeArgs::Until { stop } => *stop,
        }
    }
    /// Returns the step value of the range.
    pub fn step(&self) -> T {
        match self {
            ArangeArgs::Arange { step, .. } => *step,
            ArangeArgs::Between { .. } => T::one(),
            ArangeArgs::Until { .. } => T::one(),
        }
    }
    /// Returns the number of steps between the given boundaries
    pub fn steps(&self) -> usize where T: Real {
        match self {
            ArangeArgs::Arange { start, stop, step } => {
                let n = ((*stop - *start) / *step).ceil().to_usize().unwrap();
                n
            }
            ArangeArgs::Between { start, stop } => {
                let n = (*stop - *start).to_usize().unwrap();
                n
            }
            ArangeArgs::Until { stop } => {
                let n = stop.to_usize().unwrap();
                n
            }
        }
    }
}

impl<T> From<ops::Range<T>> for ArangeArgs<T> {
    fn from(args: ops::Range<T>) -> Self {
        ArangeArgs::Between {
            start: args.start,
            stop: args.end,
        }
    }
}



impl<T> From<ops::RangeFrom<T>> for ArangeArgs<T> {
    fn from(args: ops::RangeFrom<T>) -> Self {
        ArangeArgs::Until { stop: args.start }
    }
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
    fn from(stop: T) -> Self {
        ArangeArgs::Until { stop }
    }
}

pub trait Arange<T> {
    fn arange(args: impl Into<ArangeArgs<T>>) -> Self;
}

impl<T> Arange<T> for Vec<T>
where
    T: Float,
{
    fn arange(args: impl Into<ArangeArgs<T>>) -> Self {
        let args = args.into();
        let n: usize = args.stop().to_usize().expect("Failed to convert 'stop' to a usize");
        (0..n).map(|i| args.start() + args.step() * T::from(i).unwrap()).collect()
    }
}

impl<S, T> Arange<S> for Array<T, Ix1>
where
    S: Copy + Num + ToPrimitive,
    T: Float,
{
    fn arange(args: impl Into<ArangeArgs<S>>) -> Self {
        let args = args.into();
        let n: usize = args.stop().to_usize().expect("Failed to convert 'stop' to a usize");
        let start = T::from(args.start()).unwrap();
        let step = T::from(args.step()).unwrap();

        Array::from_iter((0..n).map(|i| start + step * T::from(i).unwrap()))
    }
}

impl<S, T> Arange<S> for Array<T, Ix2>
where
    S: Copy + Num + ToPrimitive,
    T: Float,
{
    fn arange(args: impl Into<ArangeArgs<S>>) -> Self {
        let args = args.into();
        let start = T::from(args.start()).unwrap();
        let step = T::from(args.step()).unwrap();
        let n: usize = args.stop().to_usize().expect("Failed to convert 'stop' to a usize");
        let f = | (i, _j) | {
            start + step * T::from(i).unwrap()
        };
        Array::from_shape_fn((n, 1), f)
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
