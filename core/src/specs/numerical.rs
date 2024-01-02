/*
   Appellation: num <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops::{self, Add, Div, Mul, Sub};

pub trait Algebraic<T>
where
    Self: Add<T> + Div<T> + Mul<T> + Sub<T> + Sized,
{
    type Output;
}

pub trait AlgebraicExt<T>
where
    Self: Algebraic<T>
        + ops::AddAssign<T>
        + ops::DivAssign<T>
        + ops::MulAssign<T>
        + ops::SubAssign<T>,
{
}

impl<A, B, C> Algebraic<B> for A
where
    A: Add<B, Output = C> + Div<B, Output = C> + Mul<B, Output = C> + Sub<B, Output = C>,
{
    type Output = C;
}

impl<A, B, C> AlgebraicExt<B> for A where
    A: Algebraic<B, Output = C>
        + ops::AddAssign<B>
        + ops::DivAssign<B>
        + ops::MulAssign<B>
        + ops::SubAssign<B>
{
}

pub trait ComplexNum:
    Algebraic<Self, Output = Self> + Algebraic<Self::DType, Output = Self>
{
    type DType;

    fn imag(self) -> Self::DType;

    fn real(self) -> Self::DType;
}

impl<T> ComplexNum for num::Complex<T>
where
    T: Clone + num::Num,
{
    type DType = T;

    fn imag(self) -> Self::DType {
        self.im
    }

    fn real(self) -> Self::DType {
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

pub trait Numerical<T>
where
    T: Number,
{
}
