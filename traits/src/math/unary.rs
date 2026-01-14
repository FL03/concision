/*
    Appellation: unary <traits>
    Contrib: @FL03
*/
use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Signed;

macro_rules! unary {
    (@impl $(#[$meta:meta])* $name:ident::$call:ident($($rest:tt)*)) => {
        $(#[$meta])*
        pub trait $name {
            type Output;

            fn $call($($rest)*) -> Self::Output;
        }
    };
    ($($(#[$meta:meta])*$name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(unary! { @impl $(#[$meta])* $name::$call($($rest)*) })*
    };
}

macro_rules! impl_unary_op {
    (@impl $name:ident::<$T:ty>::$method:ident) => {
        impl $name for $T {
            type Output = $T;

            fn $method(self) -> Self::Output {
                <$T>::$method(self)
            }
        }
    };
    ($($name:ident::<[$($T:ty),*]>::$method:ident),* $(,)?) => {
        $($(impl_unary_op! { @impl $name::<$T>::$method })*)*
    };
}

macro_rules! impl_something {
    (@impl $trait:ident::<$T:ty>::$method:ident($self:ident $(, $($input:ident: $I:ty),*)?) -> $out:ty {$func:expr}) => {
        impl $trait for $T {
            type Output = $out;

            fn $method($self $(, $($input: $I),*)?) -> Self::Output {
                $func
            }
        }
    };
    ($($trait:ident::<[$($T:ty),* $(,)?]>::$method:ident($self:ident) -> $out:ty {$func:expr});* $(;)?) => {
        $($(impl_something! { @impl $trait::<$T>::$method($self) -> $out {$func} } )*)*
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

impl_unary_op! {
    Abs::<[i8, i16, i32, i64, i128, isize, f32, f64]>::abs,
    Cos::<[f32, f64]>::cos,
    Cosh::<[f32, f64]>::cosh,
    Exp::<[f32, f64]>::exp,
    Sinh::<[f32, f64]>::sinh,
    Sine::<[f32, f64]>::sin,
    Tan::<[f32, f64]>::tan,
    Tanh::<[f32, f64]>::tanh,
    SquareRoot::<[f32, f64]>::sqrt
}

impl_something! {
    Squared::<[u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64]>::pow2(self) -> Self {
        self * self
    };
    Cubed::<[u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64]>::pow3(self) -> Self {
        self * self * self
    };
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
    use super::*;

    use ndarray::{Array, Dimension};
    use num_complex::{Complex, ComplexFloat};
    use num_traits::Signed;

    macro_rules! impl_complex_for {
        (@impl $name:ident::<$T:ident>::$method:ident) => {
            #[cfg(feature = "complex")]
            impl<$T> $name for num_complex::Complex<$T>
            where
                num_complex::Complex<$T>: num_complex::ComplexFloat<Real = $T>,
            {
                type Output = num_complex::Complex<$T>;

                fn $method(self) -> Self::Output {
                    num_complex::ComplexFloat::$method(self)
                }
            }
        };
        ($($name:ident::<$T:ident>::$method:ident),* $(,)?) => {
            $(impl_complex_for!(@impl $name::<$T>::$method);)*
        };
    }

    impl_complex_for! {
        Cos::<T>::cos,
        Cosh::<T>::cosh,
        Exp::<T>::exp,
        Sine::<T>::sin,
        Sinh::<T>::sinh,
        Tan::<T>::tan,
        Tanh::<T>::tanh,
        SquareRoot::<T>::sqrt,
    }

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
}
