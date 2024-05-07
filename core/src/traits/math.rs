/*
   Appellation: math <traits>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{Array, Dimension};
use num::complex::Complex;
use num::{Float, Num, Signed, Zero};

pub trait AsComplex {
    type Real;

    fn as_complex(&self, real: bool) -> Complex<Self::Real>;

    fn as_re(&self) -> Complex<Self::Real> {
        self.as_complex(true)
    }

    fn as_im(&self) -> Complex<Self::Real> {
        self.as_complex(false)
    }
}

pub trait IntoComplex: AsComplex {
    fn into_complex(self, real: bool) -> Complex<Self::Real>
    where
        Self: Sized,
    {
        self.as_complex(real)
    }

    fn into_re(self) -> Complex<Self::Real>
    where
        Self: Sized,
    {
        self.as_complex(true)
    }
}

pub trait Conjugate {
    type Output;

    fn conj(&self) -> Self::Output;
}

pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
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
    T: Copy + Num,
{
    type Real = T;

    fn as_complex(&self, real: bool) -> Complex<Self> {
        match real {
            true => Complex::new(*self, Self::zero()),
            false => Complex::new(Self::zero(), *self),
        }
    }
}

impl<T> IntoComplex for T where T: AsComplex {}

macro_rules! impl_conj {
    ($($t:ident<$res:ident>),*) => {
        $(
            impl_conj!(@impl $t<$res>);
        )*
    };
    (@impl $t:ident<$res:ident>) => {
        impl Conjugate for $t {
            type Output = $res<$t>;

            fn conj(&self) -> Self::Output {
                Complex { re: *self, im: -$t::zero() }
            }
        }
    };
}

impl_conj!(f32<Complex>, f64<Complex>);

impl<T> Conjugate for Complex<T>
where
    T: Clone + Signed,
{
    type Output = Complex<T>;

    fn conj(&self) -> Self {
        Complex::<T>::conj(self)
    }
}

impl<T, D> Conjugate for Array<T, D>
where
    D: Dimension,
    T: Clone + num::complex::ComplexFloat,
{
    type Output = Array<T, D>;
    fn conj(&self) -> Self::Output {
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
