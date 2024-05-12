/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{Array, ArrayBase, Data, Dimension};
use num::complex::{Complex, ComplexFloat};

macro_rules! unary {
    ($($name:ident::$method:ident),*) => {
        $(unary!(@impl $name::$method);)*
    };
    (@impl $name:ident::$method:ident) => {
        pub trait $name {
            type Output;

            fn $method(self) -> Self::Output;
        }
    };
    (@fn $($method:ident),* $(,)?) => {
        $(fn $method(self) -> Self::Output;)*
    };
}

unary!(Abs::abs, SquareRoot::sqrt);

/*
 ********* Implementations *********
*/
macro_rules! fwd_unop {
    ($name:ident::$method:ident<[$($T:ty),* $(,)?]>) => {
        fwd_unop!($name::$method.$method<[$($T: $T),*]>);
    };
    ($name:ident::$method:ident.$call:ident<[$($T:ty: $O:ty),* $(,)?]>) => {
        $(fwd_unop!(@impl $name::$method.$call<$T> -> $O);)*
    };
    (@impl $name:ident::$method:ident$(.$call:ident)?<$T:ty>) => {
        fwd_unop!(@impl $name::$method$(.$call)?<$T> -> $T);
    };
    (@impl $name:ident::$method:ident<$T:ty> -> $O:ty) => {
        fwd_unop!(@impl $name::$method.$method<$T> -> $O);
    };
    (@impl $name:ident::$method:ident.$call:ident<$T:ty> -> $O:ty) => {
        impl $name for $T {
            type Output = $O;

            fn $method(self) -> Self::Output {
                <$T>::$call(self)
            }
        }
    };
}

fwd_unop!(SquareRoot::sqrt<[f32, f64]>);

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
