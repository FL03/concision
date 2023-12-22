/*
   Appellation: math <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::{Float, FromPrimitive, One, Signed, Zero};
use std::ops;

pub trait Binary: One + Zero {}

impl<T> Binary for T where T: One + Zero {}

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
