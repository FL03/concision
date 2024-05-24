/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::math::Exp;
use ndarray::*;
use num::complex::{Complex, ComplexFloat};
use num::traits::Zero;

fn _relu<T>(args: T) -> T
where
    T: PartialOrd + Zero,
{
    if args > T::zero() {
        return args;
    }
    T::zero()
}

fn _sigmoid<T>(args: T) -> T
where
    T: ComplexFloat,
{
    (T::one() + args.neg().exp()).recip()
}

fn _softmax<A, S, D>(args: &ArrayBase<S, D>) -> Array<A, D>
where
    A: ComplexFloat + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    let e = args.exp();
    &e / e.sum()
}

// fn __softmax<T, I>(args: &I) -> I
// where
//     I: Clone + core::ops::Div<T, Output = I> + Exp<Output = I>, T: Exp<Output = T> + core::iter::Sum ,
//     for<'a> I: IntoIterator<Item = &'a T>,
// {
//     let e = args.exp();
//     e.clone() / e.into_iter().sum::<T>()
// }

fn _tanh<T>(args: T) -> T
where
    T: ComplexFloat,
{
    args.tanh()
}

unary!(
    ReLU::relu(self),
    Sigmoid::sigmoid(self),
    Softmax::softmax(self),
    Tanh::tanh(self),
);

/*
 ********** Implementations **********
*/
macro_rules! nonlinear {
    ($($rho:ident::$call:ident<[$($T:ty),* $(,)?]>),* $(,)? ) => {
        $(
            nonlinear!(@loop $rho::$call<[$($T),*]>);
        )*
    };
    (@loop $rho:ident::$call:ident<[$($T:ty),* $(,)?]> ) => {
        $(
            nonlinear!(@impl $rho::$call<$T>);
        )*

        nonlinear!(@arr $rho::$call);
    };
    (@impl $rho:ident::$call:ident<$T:ty>) => {
        paste::paste! {
            impl $rho for $T {
                type Output = $T;

                fn $call(self) -> Self::Output {
                    [<_ $call>](self)
                }
            }

            impl<'a> $rho for &'a $T {
                type Output = $T;

                fn $call(self) -> Self::Output {
                    [<_ $call>](*self)
                }
            }
        }


    };
    (@arr $name:ident::$call:ident) => {
        impl<A, S, D> $name for ArrayBase<S, D>
        where
            A: Clone + $name,
            D: Dimension,
            S: Data<Elem = A>
        {
            type Output = Array<<A as $name>::Output, D>;

            fn $call(self) -> Self::Output {
                self.mapv($name::$call)
            }
        }

        impl<'a, A, S, D> $name for &'a ArrayBase<S, D>
        where
            A: Clone + $name,
            D: Dimension,
            S: Data<Elem = A>
        {
            type Output = Array<<A as $name>::Output, D>;

            fn $call(self) -> Self::Output {
                self.mapv($name::$call)
            }
        }
    };
}

nonlinear!(
    ReLU::relu<[
        f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        i128,
        isize,
        u8,
        u16,
        u32,
        u64,
        u128,
        usize
    ]>,
    Sigmoid::sigmoid<[
        f32,
        f64,
        Complex<f32>,
        Complex<f64>
    ]>,
    Tanh::tanh<[
        f32,
        f64,
        Complex<f32>,
        Complex<f64>
    ]>,
);

impl<A, S, D> Softmax for ArrayBase<S, D>
where
    A: ComplexFloat + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn softmax(self) -> Self::Output {
        _softmax(&self)
    }
}

impl<'a, A, S, D> Softmax for &'a ArrayBase<S, D>
where
    A: ComplexFloat + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn softmax(self) -> Self::Output {
        _softmax(self)
    }
}
