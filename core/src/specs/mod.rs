/*
   Appellation: specs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, base::*, init::*, math::*};

pub(crate) mod arrays;
pub(crate) mod base;
pub(crate) mod init;
pub(crate) mod math;

use num::{Complex, Num, Zero};

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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_as_complex() {
        let x = 1.0;
        let y = x.as_complex();
        assert_eq!(y, Complex::new(1.0, 0.0));
    }
}
