/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Dimension, Ix2};
use num::complex::Complex;
use num::{Float, Num, Signed};

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
    T: Clone + Signed,
{
    fn conj(&self) -> Self {
        Complex::<T>::conj(self)
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

pub trait Power<Rhs> {
    type Output;

    fn pow(&self, rhs: Rhs) -> Self::Output;
}

// impl<S, T> Power<T> for S where S: Pow<T> {
//     type Output = <S as Pow<T>>::Output;

//     fn pow(self, rhs: T) -> Self::Output {
//         <Self as Pow<T>>::pow(self, rhs)
//     }
// }

impl<T> Power<usize> for Array<T, Ix2>
where
    T: Clone + Num,
    Array<T, Ix2>: Dot<Self, Output = Self>,
{
    type Output = Self;

    fn pow(&self, rhs: usize) -> Self::Output {
        if !self.is_square() {
            panic!("Matrix must be square to be raised to a power");
        }
        let mut res = Array::eye(self.shape()[0]);
        for _ in 0..rhs {
            res = res.dot(&self);
        }
        res
    }
}
