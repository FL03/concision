/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::*;
use num::complex::{Complex, ComplexFloat};
use num::traits::Zero;

pub fn relu<T>(args: &T) -> T
where
    T: Clone + PartialOrd + Zero,
{
    if args > &T::zero() {
        return args.clone();
    }
    T::zero()
}

pub fn sigmoid<T>(args: &T) -> T
where
    T: ComplexFloat,
{
    (T::one() + (*args).neg().exp()).recip()
}

pub fn softmax<A, S, D>(args: &ArrayBase<S, D>) -> Array<A, D>
where
    A: ComplexFloat,
    D: Dimension,
    S: Data<Elem = A>,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub fn softmax_axis<T, D>(args: &Array<T, D>, axis: Option<usize>) -> Array<T, D>
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    let exp = args.mapv(|x| x.exp());
    if let Some(axis) = axis {
        let denom = exp.sum_axis(Axis(axis));
        exp / denom
    } else {
        let denom = exp.sum();
        exp / denom
    }
}

pub fn tanh<T>(args: &T) -> T
where
    T: ComplexFloat,
{
    args.tanh()
}

unary!(
    ReLU::relu(&self),
    Sigmoid::sigmoid(&self),
    Softmax::softmax(&self),
    Tanh::tanh(&self),
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
        impl $rho for $T {
            type Output = $T;

            fn $call(&self) -> Self::Output {
                $call(self)
            }
        }

        impl<'a> $rho for &'a $T {
            type Output = $T;

            fn $call(&self) -> Self::Output {
                $call(*self)
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

            fn $call(&self) -> Self::Output {
                self.map($name::$call)
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
        Complex < f64 >
    ]>,
);

impl<A, S, D> Softmax for ArrayBase<S, D>
where
    A: ComplexFloat,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn softmax(&self) -> Self::Output {
        softmax(self)
    }
}
