/*
   Appellation: specs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, base::*, init::*, math::*, numerical::*};

pub(crate) mod arrays;
pub(crate) mod base;
pub(crate) mod init;
pub(crate) mod math;
pub(crate) mod numerical;

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
        let (re, im): (Self, Self) = if real {
            (self, Self::zero())
        } else {
            (Self::zero(), self)
        };
        Complex::new(re, im)
    }
}

pub trait Named {
    fn name(&self) -> &str;
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
    use ndarray::prelude::*;

    #[test]
    fn test_arange() {
        let exp = array![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(&exp, &Array1::<f64>::arange(5))
    }

    #[test]
    fn test_as_complex() {
        let x = 1.0;
        let y = x.as_re();
        assert_eq!(y, Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_affine() {
        let x = array![[0.0, 1.0], [2.0, 3.0]];

        let y = x.affine(4.0, -2.0).unwrap();
        assert_eq!(y, array![[-2.0, 2.0], [6.0, 10.0]]);
    }
}
