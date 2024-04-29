/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::linalg::Dot;
use nd::{Array, Dimension, Ix2};
use num::complex::Complex;
use num::{Float, Num, Signed};

pub trait AsComplex: Sized {
    fn as_complex(self, real: bool) -> Complex<Self>;

    fn as_re(self) -> Complex<Self> {
        self.as_complex(true)
    }

    fn as_im(self) -> Complex<Self> {
        self.as_complex(false)
    }
}

pub trait Conjugate {
    fn conj(&self) -> Self;
}

pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
}

pub trait Pow<Rhs = Self> {
    type Output;

    fn pow(&self, rhs: Rhs) -> Self::Output;
}

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

pub trait SquareRoot {
    fn sqrt(self) -> Self;
}

/*
 ********* Implementations *********
*/
impl<T> AsComplex for T
where
    T: Num,
{
    fn as_complex(self, real: bool) -> Complex<Self> {
        match real {
            true => Complex::new(self, Self::zero()),
            false => Complex::new(Self::zero(), self),
        }
    }
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

impl<T, D> Conjugate for Array<T, D>
where
    D: Dimension,
    T: Clone + Conjugate,
{
    fn conj(&self) -> Self {
        self.mapv(|x| x.conj())
    }
}

impl<T> FloorDiv for T
where
    T: Copy + Num,
{
    type Output = T;

    fn floor_div(self, rhs: Self) -> Self::Output {
        crate::floor_div(self, rhs)
    }
}

// impl<A, B, C> Pow<B> for A where A: num::traits::Pow<B, Output = C> {
//     type Output = C;

//     fn pow(self, rhs: B) -> Self::Output {
//         num::traits::Pow::pow(self, rhs)
//     }
// }

// impl<D, A, B, C> Pow<i32> for Array<A, D> where A: Clone + num::traits::Pow<B, Output = C>, D: Dimension  {
//     type Output = Array<C, D>;

//     fn pow(self, rhs: B) -> Self::Output {
//         self.mapv(|x| x.pow(rhs))
//     }
// }

impl<T> Pow<i32> for Array<T, Ix2>
where
    T: Clone + Num,
    Array<T, Ix2>: Dot<Self, Output = Self>,
{
    type Output = Self;

    fn pow(&self, rhs: i32) -> Self::Output {
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

impl<T> RoundTo for T
where
    T: Float,
{
    fn round_to(&self, places: usize) -> Self {
        crate::round_to(*self, places)
    }
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
