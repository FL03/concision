/*
   Appellation: elements <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::complex::Complex;
use num::traits::real::Real;
use num::traits::{NumAssign, NumCast, NumOps};
use std::fmt::{LowerExp, UpperExp};
use std::iter::{Product, Sum};
use std::ops::Neg;

pub trait UPLOExp: LowerExp + UpperExp {}

pub trait Element: NumCast + NumOps + NumOps<Complex<Self>, Complex<Self>> {}

pub trait Scalar:
    Copy + Neg<Output = Self> + NumAssign + NumCast + NumOps + Product<Self> + Sum<Self> + 'static
{
    type Complex: Scalar<Complex = Self::Complex, Real = Self::Real>
        + NumOps<<Self as Scalar>::Real>;
    type Real: Real
        + Scalar<Complex = Self::Complex, Real = Self::Real>
        + NumOps<<Self as Scalar>::Complex, <Self as Scalar>::Complex>;
}
