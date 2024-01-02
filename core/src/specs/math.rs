/*
   Appellation: math <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2};
use num::complex::Complex;
use num::{Float, Num, Signed};
use std::ops;

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
    T: Clone + Num + Signed,
{
    fn conj(&self) -> Self {
        Complex::<T>::new(self.re.clone(), -self.im.clone())
    }
}

// impl<T> Conjugate for T
// where
//     T: ComplexFloat,
// {
//     fn conj(&self) -> Self {
//         ComplexFloat::conj(self)
//     }
// }

impl<T, D> Conjugate for Array<T, D>
where
    D: Dimension,
    T: Clone + Conjugate,
{
    fn conj(&self) -> Self {
        self.mapv(|x| x.conj())
    }
}

pub trait Arithmetic<S>
where
    Self: ops::Add<S> + ops::Div<S> + ops::Mul<S> + ops::Sub<S>,
{
    type Output;
}

impl<A, S, T> Arithmetic<S> for A
where
    A: ops::Add<S, Output = T>
        + ops::Div<S, Output = T>
        + ops::Mul<S, Output = T>
        + ops::Sub<S, Output = T>,
{
    type Output = T;
}

pub trait MatrixOps<T = f64, A = Ix2, B = Ix2>:
    Arithmetic<Array<T, A>, Output = Array<T, B>> + Sized
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
    T: Arithmetic<T>,
    Self: Arithmetic<Array<T, A>, Output = Array<T, B>>,
{
}

impl<T, D, A, B> MatrixOps<T, A, B> for &Array<T, D>
where
    A: Dimension,
    B: Dimension,
    D: Dimension,
    T: Arithmetic<T>,
    Self: Arithmetic<Array<T, A>, Output = Array<T, B>>,
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

impl<T, D> SquareRoot for Array<T, D>
where
    D: Dimension,
    T: Float,
{
    fn sqrt(self) -> Self {
        self.mapv(|x| x.sqrt())
    }
}
