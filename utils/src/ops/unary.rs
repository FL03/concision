/*
    Appellation: unary <traits>
    Contrib: @FL03
*/
use ndarray::{Array, ArrayBase, Data, Dimension};
use num::complex::{Complex, ComplexFloat};
use num::traits::Signed;

unary! {
    Abs::abs(self),
    Cos::cos(self),
    Cosh::cosh(self),
    Exp::exp(self),
    Sine::sin(self),
    Sinh::sinh(self),
    Tan::tan(self),
    Tanh::tanh(self),
    Squared::pow2(self),
    Cubed::pow3(self),
    SquareRoot::sqrt(self)
}

unary! {
    Conjugate::conj(&self),
}

/*
 ********* Implementations *********
*/

macro_rules! unary_impl {
    ($($name:ident<$T:ty$(, Output = $O:ty)?>::$method:ident),* $(,)?) => {
        $(unary_impl!(@impl $name::$method<$T$(, Output = $O>)?);)*
    };
    ($($name:ident::<$T:ty, Output = $O:ty>::$method:ident),* $(,)?) => {
        $(unary_impl!(@impl $name::$method<$T, Output = $O>);)*
    };
    ($($name:ident::<[$($T:ty),*]>::$method:ident),* $(,)?) => {
        $(unary_impl!(@loop $name::$method<[$($T),*]>);)*
    };
    (@loop $name:ident::<[$($T:ty),* $(,)?]>::$method:ident) => {
        $(unary_impl!(@impl $name::<$T>::$method);)*
    };
    (@impl $name:ident::<$T:ty>::$method:ident) => {
        unary_impl!(@impl $name::<$T, Output = $T>::$method);
    };
    (@impl $name:ident::<$T:ty, Output = $O:ty>::$method:ident) => {
        impl $name for $T {
            type Output = $O;

            fn $method(self) -> Self::Output {
                <$T>::$method(self)
            }
        }
    };
}

macro_rules! unary_impls {
    ($($name:ident::<[$($T:ty),* $(,)?]>::$method:ident),* $(,)?) => {
        $(unary_impl!(@loop $name::<[$($T),*]>::$method);)*
    };
}

unary_impls! {
    Abs::<[f32, f64]>::abs,
    Cos::<[f32, f64, Complex<f32>, Complex<f64>]>::cos,
    Cosh::<[f32, f64, Complex<f32>, Complex<f64>]>::cosh,
    Exp::<[f32, f64, Complex<f32>, Complex<f64>]>::exp,
    Sinh::<[f32, f64, Complex<f32>, Complex<f64>]>::sinh,
    Sine::<[f32, f64, Complex<f32>, Complex<f64>]>::sin,
    Tan::<[f32, f64, Complex<f32>, Complex<f64>]>::tan,
    Tanh::<[f32, f64, Complex<f32>, Complex<f64>]>::tanh,
    SquareRoot::<[f32, f64]>::sqrt
}

/*
 ************* macro implementations *************
*/

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
                Complex { re: *self, im: num_traits::Zero::zero() }
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

impl<A, S, D> Abs for ArrayBase<S, D>
where
    A: Clone + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn abs(self) -> Self::Output {
        self.mapv(|x| x.abs())
    }
}

impl<A, S, D> Abs for &ArrayBase<S, D>
where
    A: Clone + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn abs(self) -> Self::Output {
        self.mapv(|x| x.abs())
    }
}

impl<A> Squared for A
where
    A: Clone + core::ops::Mul<Output = A>,
{
    type Output = A;

    fn pow2(self) -> Self::Output {
        self.clone() * self
    }
}

impl<A> SquareRoot for Complex<A>
where
    Complex<A>: ComplexFloat<Real = A>,
{
    type Output = Self;

    fn sqrt(self) -> Self::Output {
        ComplexFloat::sqrt(self)
    }
}

impl<A, B, S, D> SquareRoot for ArrayBase<S, D>
where
    A: Clone + SquareRoot<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn sqrt(self) -> Self::Output {
        self.mapv(|x| x.sqrt())
    }
}

impl<A, B, S, D> Exp for ArrayBase<S, D>
where
    A: Clone + Exp<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn exp(self) -> Self::Output {
        self.mapv(|x| x.exp())
    }
}

impl<A, S, D> Exp for &ArrayBase<S, D>
where
    A: Clone + ComplexFloat,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn exp(self) -> Self::Output {
        self.mapv(|x| x.exp())
    }
}
