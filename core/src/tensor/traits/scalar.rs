/*
    Appellation: scalar <module>
    Contrib: @FL03
*/
/// [Numerical] is a trait for all numerical types; implements a number of core operations
pub trait Numerical
where
    Self: Clone
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
        + core::ops::SubAssign,
{
    private!();
}

/// The [Scalar] trait extends the [Numerical] trait to include additional mathematical
/// operations for the purpose of reducing the number of overall traits required to
/// complete various machine-learning tasks.
pub trait Scalar
where
    Self: Numerical
        + Sized
        + 'static
        + core::fmt::Display
        + core::iter::Product
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
        + num_traits::Signed
        + num_traits::Pow<Self, Output = Self>
        + num_traits::Float
        + num_traits::FloatConst,
{
    private!();

    fn one() -> Self
    where
        Self: Sized,
    {
        num_traits::One::one()
    }

    fn zero() -> Self
    where
        Self: Sized,
    {
        num_traits::Zero::zero()
    }
}

#[cfg(feature = "complex")]
pub trait ScalarComplex
where
    Self::Complex<Self::Real>: num::complex::ComplexFloat,
{
    type Real: num_traits::real::Real;
    type Complex<T>;

    private!();
    /// create a new complex number
    fn new(real: Self::Real, imag: Self::Real) -> Self::Complex<Self::Real>;
    /// returns a reference to the real part of the object
    fn real(&self) -> Self::Real;
    /// returns a reference to the imaginary part of the object
    fn imag(&self) -> Self::Real;
    /// returns the absolute value of the complex number
    fn abs(&self) -> Self::Real {
        use num_traits::real::Real;
        self.real().hypot(self.imag())
    }
    /// compute the complex conjugate of the object
    fn conj(&self) -> Self::Complex<Self::Real> {
        use core::ops::Neg;
        Self::new(self.real(), self.imag().neg())
    }
}

#[cfg(feature = "complex")]
impl<U> ScalarComplex for U
where
    U: num::complex::ComplexFloat,
    U::Real: Scalar,
{
    type Real = U::Real;
    type Complex<V> = num_complex::Complex<V>;

    seal!();

    fn new(real: Self::Real, imag: Self::Real) -> Self::Complex<Self::Real> {
        num_complex::Complex::new(real, imag)
    }

    fn real(&self) -> Self::Real {
        self.re()
    }

    fn imag(&self) -> Self::Real {
        self.im()
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
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize
}

impl_scalar! {
    #[scalar]
    f32,
    f64,
}

impl<A, S, D> Numerical for ndarray::ArrayBase<S, D>
where
    A: Numerical,
    D: ndarray::Dimension,
    S: ndarray::RawData<Elem = A> + ndarray::Data + ndarray::DataMut + ndarray::DataOwned,
    ndarray::ArrayBase<S, D>: Numerical,
{
    seal!();
}
