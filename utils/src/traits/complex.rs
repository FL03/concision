/*
   Appellation: num <traits>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "complex")]
use num::Num;
use num::complex::Complex;
pub trait AsComplex<T> {
    type Complex<A>;

    fn as_complex(&self, real: bool) -> Self::Complex<T>;

    fn as_re(&self) -> Self::Complex<T> {
        self.as_complex(true)
    }

    fn as_im(&self) -> Self::Complex<T> {
        self.as_complex(false)
    }
}
/// Trait for converting a type into a complex number.
pub trait IntoComplex<T> {
    type Complex<A>;

    fn into_complex(self, real: bool) -> Self::Complex<T>
    where
        Self: Sized;

    fn into_re(self) -> Self::Complex<T>
    where
        Self: Sized,
    {
        self.into_complex(true)
    }

    fn into_im(self) -> Self::Complex<T>
    where
        Self: Sized,
    {
        self.into_complex(false)
    }
}

/*
 ********* Implementations *********
*/

impl<T> AsComplex<T> for T
where
    T: Clone + Num,
{
    type Complex<A> = Complex<A>;

    fn as_complex(&self, real: bool) -> Complex<T> {
        match real {
            true => Complex::new(self.clone(), Self::zero()),
            false => Complex::new(Self::zero(), self.clone()),
        }
    }
}

impl<T> IntoComplex<T> for T
where
    T: Num,
{
    type Complex<A> = Complex<A>;

    fn into_complex(self, real: bool) -> Self::Complex<T>
    where
        Self: Sized,
    {
        match real {
            true => Complex::new(self, T::zero()),
            false => Complex::new(T::zero(), self),
        }
    }
}
