/*
   Appellation: num <traits>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "complex")]

use num_complex::Complex;
use num_traits::Num;

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
    /// uses the current state as the real value of a complex number
    fn into_re(self) -> Self::Complex<T>
    where
        Self: Sized,
    {
        self.into_complex(true)
    }
    /// uses the current state as the imaginary value of a complex number
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
    T: Clone + IntoComplex<T>,
{
    type Complex<A> = <T as IntoComplex<T>>::Complex<A>;

    fn as_complex(&self, real: bool) -> Self::Complex<T> {
        self.clone().into_complex(real)
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
        if real {
            Complex::new(self, T::zero())
        } else {
            Complex::new(T::zero(), self)
        }
    }
}
