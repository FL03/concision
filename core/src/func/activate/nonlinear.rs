/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::utils::*;
use nd::*;
use num::complex::{Complex, ComplexFloat};
use num::traits::Zero;

unary!(
    ReLU::relu(self),
    Sigmoid::sigmoid(self),
    Softmax::softmax(self),
    Tanh::tanh(self),
);

pub trait SoftmaxAxis: Softmax {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}

pub trait NonLinear {
    type Output;

    fn relu(self) -> Self::Output;
    fn sigmoid(self) -> Self::Output;
    fn softmax(self) -> Self::Output;
    fn softmax_axis(self, axis: usize) -> Self::Output;
    fn tanh(self) -> Self::Output;
}

/*
 ********** Implementations **********
*/
macro_rules! nonlinear {
    ($($rho:ident::$call:ident<[$($T:ty),* $(,)?]>($f:expr)),* $(,)? ) => {
        $(
            nonlinear!(@loop $rho::$call<[$($T),*]>($f));
        )*
    };
    (@loop $rho:ident::$call:ident<[$($T:ty),* $(,)?]>($f:expr) ) => {
        $(
            nonlinear!(@impl $rho::$call<$T>($f));
        )*
    };
    (@impl $rho:ident::$call:ident<$T:ty>($f:expr)) => {
        impl $rho for $T {
            type Output = $T;

            fn $call(self) -> Self::Output {
                $f(self)
            }
        }

        impl<'a> $rho for &'a $T {
            type Output = $T;

            fn $call(self) -> Self::Output {
                $f(*self)
            }
        }
    };
}

macro_rules! nonlinear_rho {
    ($name:ident::$call:ident where A: $($rest:tt)* ) => {
        impl<A, S, D> $name for ArrayBase<S, D>
        where
            D: Dimension,
            S: Data<Elem = A>,
            A: $($rest)*
        {
            type Output = Array<A, D>;

            fn $call(self) -> Self::Output {
                self.mapv($call)
            }
        }

        impl<'a, A, S, D> $name for &'a ArrayBase<S, D>
        where
            D: Dimension,
            S: Data<Elem = A>,
            A: $($rest)*
        {
            type Output = Array<A, D>;

            fn $call(self) -> Self::Output {
                self.mapv($call)
            }
        }
    };
    (alt $name:ident::$call:ident where A: $($rest:tt)* ) => {
        impl<A, S, D> $name for ArrayBase<S, D>
        where
            D: Dimension,
            S: Data<Elem = A>,
            A: $($rest)*
        {
            type Output = Array<A, D>;

            fn $call(self) -> Self::Output {
                $call(&self)
            }
        }

        impl<'a, A, S, D> $name for &'a ArrayBase<S, D>
        where
            D: Dimension,
            S: Data<Elem = A>,
            A: $($rest)*
        {
            type Output = Array<A, D>;

            fn $call(self) -> Self::Output {
                $call(self)
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
    ]>(relu),
    Sigmoid::sigmoid<[
        f32,
        f64,
        Complex<f32>,
        Complex<f64>
    ]>(sigmoid),
    Tanh::tanh<[
        f32,
        f64,
        Complex<f32>,
        Complex<f64>
    ]>(tanh),
);

nonlinear_rho!(ReLU::relu where A: Clone + PartialOrd + Zero);
nonlinear_rho!(Sigmoid::sigmoid where A: ComplexFloat);
nonlinear_rho!(alt Softmax::softmax where A: ComplexFloat + ScalarOperand);
nonlinear_rho!(Tanh::tanh where A: ComplexFloat);

impl<A, S, D> SoftmaxAxis for ArrayBase<S, D>
where
    A: ComplexFloat + ScalarOperand,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    fn softmax_axis(self, axis: usize) -> Self::Output {
        softmax_axis(&self, axis)
    }
}

impl<'a, A, S, D> SoftmaxAxis for &'a ArrayBase<S, D>
where
    A: ComplexFloat + ScalarOperand,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    fn softmax_axis(self, axis: usize) -> Self::Output {
        softmax_axis(&self, axis)
    }
}
