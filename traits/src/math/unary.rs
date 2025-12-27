/*
    Appellation: unary <traits>
    Contrib: @FL03
*/
use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Signed;

macro_rules! unary {
    (@impl $name:ident::$call:ident($($rest:tt)*)) => {
        pub trait $name {
            type Output;

            fn $call($($rest)*) -> Self::Output;
        }
    };
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
    };
}

macro_rules! unary_impl {
    (@impl $name:ident::<$T:ty>::$method:ident) => {
        impl $name for $T {
            type Output = $T;

            fn $method(self) -> Self::Output {
                <$T>::$method(self)
            }
        }
    };
    (@impl #[complex] $name:ident::<$T:ty>::$method:ident) => {
        #[cfg(feature = "complex")]
        impl $name for num_complex::Complex<$T>
        where
            num_complex::Complex<$T>: num_complex::ComplexFloat<Real = $T>,
        {
            type Output = num_complex::Complex<$T>;

            fn $method(self) -> Self::Output {
                num_complex::ComplexFloat::$method(self)
            }
        }
    };
    ($($name:ident::<[$($T:ty),*]>::$method:ident),* $(,)?) => {
        $($(unary_impl!(@impl $name::<$T>::$method);)*)*
    };
}

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
    SquareRoot::sqrt(self),
    Conjugate::conj(&self),
}

/*
 ********* Implementations *********
*/

unary_impl! {
    Abs::<[f32, f64]>::abs,
    Cos::<[f32, f64]>::cos,
    Cosh::<[f32, f64]>::cosh,
    Exp::<[f32, f64]>::exp,
    Sinh::<[f32, f64]>::sinh,
    Sine::<[f32, f64]>::sin,
    Tan::<[f32, f64]>::tan,
    Tanh::<[f32, f64]>::tanh,
    SquareRoot::<[f32, f64]>::sqrt
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
impl<A, S, D> Exp for &ArrayBase<S, D, A>
where
    A: Clone + Exp<Output = A>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn exp(self) -> Self::Output {
        self.mapv(|x| x.exp())
    }
}

#[cfg(feature = "complex")]
mod impl_complex {
    use super::{Conjugate, Cos, Exp, SquareRoot};

    use ndarray::{Array, Dimension};
    use num_complex::{Complex, ComplexFloat};
    use num_traits::Signed;

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
        T: Clone + ComplexFloat,
    {
        type Output = Array<T, D>;
        fn conj(&self) -> Self::Output {
            self.mapv(|x| x.conj())
        }
    }

    impl<T> Cos for Complex<T>
    where
        Complex<T>: ComplexFloat,
    {
        type Output = Self;

        fn cos(self) -> Self::Output {
            ComplexFloat::cos(self)
        }
    }

    impl<T> Exp for Complex<T>
    where
        Complex<T>: ComplexFloat,
    {
        type Output = Self;

        fn exp(self) -> Self::Output {
            ComplexFloat::exp(self)
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
}
