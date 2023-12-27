/*
   Appellation: math <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::{Complex, Float, FromPrimitive, Num, One, Signed, Zero};
use std::ops;

pub trait Binary: One + Zero {}

impl<T> Binary for T where T: One + Zero {}

pub trait Conjugate {
    fn conj(&self) -> Self;
}

impl Conjugate for f32 {
    fn conj(&self) -> Self {
        *self
    }
}

impl Conjugate for f64 {
    fn conj(&self) -> Self {
        *self
    }
}

impl<T> Conjugate for Complex<T>
where
    T: Copy + Num + Signed,
{
    fn conj(&self) -> Self {
        Complex::<T>::new(self.re, -self.im)
    }
}

impl<T, D> Conjugate for Array<T, D>
where
    D: Dimension,
    T: Clone + Conjugate,
{
    fn conj(&self) -> Self {
        self.mapv(|x| x.conj())
    }
}

pub trait FloatExt: FromPrimitive + NdFloat + Signed + SampleUniform {}

impl<T> FloatExt for T where T: FromPrimitive + NdFloat + Signed + SampleUniform {}

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

pub trait SquareRoot {
    fn sqrt(self) -> Self;
}

impl SquareRoot for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl SquareRoot for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

impl<T> SquareRoot for Complex<T>
where
    T: Float,
{
    fn sqrt(self) -> Self {
        Complex::<T>::sqrt(self)
    }
}
