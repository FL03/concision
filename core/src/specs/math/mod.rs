/*
   Appellation: math <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{numerical::*, ops::*};

pub(crate) mod numerical;
pub(crate) mod ops;

use num::complex::Complex;
use num::traits::{Float, Num};

pub trait AsComplex: Sized {
    fn as_complex(self, real: bool) -> Complex<Self>;

    fn as_re(self) -> Complex<Self> {
        self.as_complex(true)
    }

    fn as_im(self) -> Complex<Self> {
        self.as_complex(false)
    }
}

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

pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
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

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

impl<T> RoundTo for T
where
    T: Float,
{
    fn round_to(&self, places: usize) -> Self {
        crate::round_to(*self, places)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_complex() {
        let x = 1.0;
        let y = x.as_re();
        assert_eq!(y, Complex::new(1.0, 0.0));
    }
}
