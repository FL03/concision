/*
    Appellation: scalar <module>
    Contrib: @FL03
*/
use num_complex::Complex;

pub trait Numerical:
    Clone
    + Copy
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + core::fmt::Debug
    + core::ops::Add
    + core::ops::AddAssign
    + core::ops::Div
    + core::ops::DivAssign
    + core::ops::Mul
    + core::ops::MulAssign
    + core::ops::Rem
    + core::ops::RemAssign
    + core::ops::Sub
    + core::ops::SubAssign
{
    private!();
}

pub trait Scalar
where
    Self: Numerical
        + 'static
        + core::fmt::Display
        + core::iter::Sum
        + core::ops::Neg
        + num_traits::One
        + num_traits::Zero
        + num_traits::Num
        + num_traits::NumCast
        + num_traits::NumAssign
        + num_traits::NumAssignOps
        + num_traits::NumAssignRef
        + num_traits::NumOps
        + num_traits::NumRef
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + num_traits::Signed,
{
    private!();
}

#[cfg(feature = "complex")]
pub trait ScalarComplex {
    type Real: Scalar;
    type Complex<T>
    where
        T: Scalar;

    private!();
    /// create a new complex number
    fn new(real: Self::Real, imag: Self::Real) -> Self::Complex<Self::Real>;
    /// returns a reference to the real part of the object
    fn real(&self) -> Self::Real;
    /// returns a mutable reference to the real part of the object
    fn real_mut(&mut self) -> &mut Self::Real;
    /// returns a reference to the imaginary part of the object
    fn imag(&self) -> Self::Real;
    /// returns a mutable reference to the imaginary part of the object
    fn imag_mut(&mut self) -> &mut Self::Real;
    /// compute the complex conjugate of the object
    fn conj(&self) -> Self::Complex<Self::Real> {
        use core::ops::Neg;
        Self::new(self.real(), self.imag().neg())
    }
    /// replace the imaginary value with another and return the old value
    fn replace_imag(&mut self, imag: Self::Real) -> Self::Real {
        core::mem::replace(self.imag_mut(), imag)
    }
    /// replace the real value with another and return the old value
    fn replace_real(&mut self, real: Self::Real) -> Self::Real {
        core::mem::replace(self.real_mut(), real)
    }
    /// update the imaginary value
    fn set_imag(&mut self, imag: Self::Real) -> &mut Self {
        *self.imag_mut() = imag;
        self
    }
    /// update the real value
    fn set_real(&mut self, real: Self::Real) -> &mut Self {
        *self.real_mut() = real;
        self
    }
}

#[cfg(feature = "complex")]
impl<U> ScalarComplex for Complex<U>
where
    U: Scalar,
{
    type Real = U;
    type Complex<V: Scalar> = Complex<V>;

    seal!();

    fn new(real: Self::Real, imag: Self::Real) -> Self::Complex<Self::Real> {
        Complex::new(real, imag)
    }

    fn real(&self) -> Self::Real {
        self.re
    }

    fn real_mut(&mut self) -> &mut Self::Real {
        &mut self.re
    }

    fn imag(&self) -> Self::Real {
        self.im
    }

    fn imag_mut(&mut self) -> &mut Self::Real {
        &mut self.im
    }
}

macro_rules! impl_scalar {
    (@impl $t:ty) => {
        impl Numerical for $t {
            seal!();
        }
    };
    (@scalar $t:ty) => {
        impl_scalar!(@impl $t);

        impl Scalar for $t {
            seal!();
        }
    };
    ($($t:ty),* $(,)?) => {
        $(
            impl_scalar!(@impl $t);
        )*
    };
    (#[scalar] $($t:ty),* $(,)?) => {
        $(
            impl_scalar!(@scalar $t);
        )*
    };
}

impl_scalar! {
    u8,
    u16,
    u32,
    u64,
    u128,
    usize
}

impl_scalar! {
    #[scalar]
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
}

impl<A, S, D> Numerical for ndarray::ArrayBase<S, D>
where
    A: Numerical + Scalar,
    S: ndarray::DataOwned<Elem = A>,
    D: ndarray::Dimension,
    ndarray::ArrayBase<S, D>: Numerical,
{
    seal!();
}
