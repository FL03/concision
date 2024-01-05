/*
   Appellation: num <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::complex::Complex;
use num::traits::{NumAssignOps, NumOps, Signed};
use std::ops::{Add, Div, Mul, Sub};

pub trait Algebraic<T>
where
    Self: Add<T> + Div<T> + Mul<T> + Sub<T> + Sized,
{
    type Output;
}

pub trait AlgebraicExt<T>
where
    Self: Algebraic<T> + NumAssignOps,
{
}

impl<A, B, C> Algebraic<B> for A
where
    A: Add<B, Output = C> + Div<B, Output = C> + Mul<B, Output = C> + Sub<B, Output = C>,
{
    type Output = C;
}

impl<A, B, C> AlgebraicExt<B> for A where A: Algebraic<B, Output = C> + NumAssignOps {}

pub trait ComplexNum
where
    Self: Algebraic<Self, Output = Self> + Algebraic<Self::DType, Output = Self>,
{
    type DType;

    fn complex(self) -> Self;

    fn im(self) -> Self::DType;

    fn re(self) -> Self::DType;
}

pub trait Imaginary:
    Algebraic<Self, Output = Self> + Algebraic<Self::DType, Output = Self>
{
    type DType;

    fn im(self) -> Self::DType;

    fn re(self) -> Self::DType;
}

impl<T> Imaginary for num::Complex<T>
where
    T: Clone + num::Num,
{
    type DType = T;

    fn im(self) -> Self::DType {
        self.im
    }

    fn re(self) -> Self::DType {
        self.re
    }
}

pub trait Number {}

impl Number for i8 {}

impl Number for i16 {}

impl Number for i32 {}

impl Number for i64 {}

impl Number for i128 {}

impl Number for isize {}

impl Number for u8 {}

impl Number for u16 {}

impl Number for u32 {}

impl Number for u64 {}

impl Number for u128 {}

impl Number for usize {}

impl Number for f32 {}

impl Number for f64 {}

impl<S, T> Number for S where S: ComplexNum<DType = T> {}

pub trait Abs {
    fn abs(&self) -> Self;
}

impl<T> Abs for T
where
    T: Signed,
{
    fn abs(&self) -> Self {
        Signed::abs(self)
    }
}
pub trait Scalar {
    type Complex: NumOps + NumOps<Self::Real, Self::Complex>;
    type Real: NumOps + NumOps<Self::Complex, Self::Complex>;
}

pub trait Numerical: Algebraic<Self, Output = Self> {
    type Elem: Algebraic<Self::Elem, Output = Self::Elem> + Number;

    fn abs(&self) -> Self
    where
        Self::Elem: Abs,
    {
        self.eval(|x| x.abs())
    }

    fn conj(self) -> Self;

    fn eval<F>(&self, f: F) -> Self
    where
        F: Fn(Self::Elem) -> Self::Elem;
}
