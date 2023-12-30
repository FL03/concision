/*
   Appellation: specs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, base::*, init::*, math::*};

pub(crate) mod arrays;
pub(crate) mod base;
pub(crate) mod init;
pub(crate) mod math;

use num::traits::float::FloatCore;
use num::{Complex, Num, Zero};

pub trait CncFloat: FloatCore {}

impl<T> CncFloat for T where T: FloatCore {}

pub trait AsComplex: Num {
    fn as_complex(&self) -> Complex<Self>;

    fn as_imag(&self) -> Complex<Self>;
}

impl<T> AsComplex for T
where
    T: Copy + Num + Zero,
{
    fn as_complex(&self) -> Complex<Self> {
        Complex::new(*self, T::zero())
    }

    fn as_imag(&self) -> Complex<Self> {
        Complex::new(T::zero(), *self)
    }
}

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

impl<T> RoundTo for T
where
    T: num::Float,
{
    fn round_to(&self, places: usize) -> Self {
        crate::round_to(*self, places)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_as_complex() {
        let x = 1.0;
        let y = x.as_complex();
        assert_eq!(y, Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_affine() {
        let x = array![[0.0, 1.0], [2.0, 3.0]];

        let y = x.affine(4.0, -2.0).unwrap();
        assert_eq!(y, array![[-2.0, 2.0], [6.0, 10.0]]);
    }
}
