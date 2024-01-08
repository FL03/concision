/*
   Appellation: num <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::complex::Complex;
use num::traits::{Num, NumAssignOps, NumOps, Signed};
// use num::traits::real::Real;

pub trait Algebraic<S = Self, T = Self>: NumOps<S, T> + Sized {}

pub trait AlgebraicExt<S, T>
where
    Self: Algebraic<S, T> + NumAssignOps + Sized,
{
}

impl<A, B, C> Algebraic<B, C> for A where A: NumOps<B, C> + Sized {}

pub trait ComplexNum: Sized {
    type Real: Algebraic + Algebraic<Self, Self>;

    fn complex(self) -> Self;

    fn im(self) -> Self::Real;

    fn re(self) -> Self::Real;
}

pub trait Imaginary<T>: Sized
where
    T: Algebraic + Algebraic<Self::Complex, Self::Complex>,
{
    type Complex: Algebraic + Algebraic<T>;

    fn im(self) -> T;

    fn re(self) -> T;
}

impl<T> Imaginary<T> for Complex<T>
where
    T: Algebraic + Algebraic<Complex<T>, Complex<T>> + Clone + Num,
{
    type Complex = Complex<T>;

    fn im(self) -> T {
        self.im
    }

    fn re(self) -> T {
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

impl<S, T> Number for S where S: ComplexNum<Real = T> {}

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

pub trait Reciprocal {
    fn recip(self) -> Self;
}

impl<T> Reciprocal for T
where
    T: Num + NumOps,
{
    fn recip(self) -> Self {
        Self::one() / self
    }
}

pub trait Numerical: Sized {
    type Elem: Algebraic + Number;

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
