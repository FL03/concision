/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, Axis, Dimension, NdFloat, RemoveAxis};
use num::complex::ComplexFloat;
use num::{Float, One, Zero};

pub fn relu<T>(args: &T) -> T
where
    T: Clone + PartialOrd + Zero,
{
    if args > &T::zero() {
        args.clone()
    } else {
        T::zero()
    }
}

pub fn sigmoid<T, D>(args: &Array<T, D>) -> Array<<T as Sigmoid>::Output, D>
where
    D: Dimension,
    T: Clone + Sigmoid,
{
    args.mapv(|x| x.sigmoid())
}

pub fn softmax<T, D>(args: &Array<T, D>) -> Array<T, D>
where
    D: Dimension,
    T: Float,
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

pub fn tanh<T, D>(args: &Array<T, D>) -> Array<<T as Tanh>::Output, D>
where
    D: Dimension,
    T: Clone + Tanh,
{
    args.mapv(|x| x.tanh())
}

macro_rules! unary {
    ($($name:ident.$call:ident),* $(,)?) => {
        $(
            unary!(@impl $name.$call);
        )*
    };
    (@impl $name:ident.$call:ident) => {
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
}

unary!(ReLU.relu, Sigmoid.sigmoid, Softmax.softmax, Tanh.tanh,);

/*
 ********** Implementations **********
*/

impl<T, D> Sigmoid for Array<T, D>
where
    D: Dimension,
    T: Clone + Sigmoid,
{
    type Output = Array<<T as Sigmoid>::Output, D>;

    fn sigmoid(&self) -> Self::Output {
        self.mapv(|x| x.sigmoid())
    }
}

macro_rules! impl_sigmoid {
    ($($T:ty),* $(,)?) => {
        $(
            impl_sigmoid!(@base $T);
        )*
    };
    (@base $T:ty) => {
        impl Sigmoid for $T {
            type Output = $T;

            fn sigmoid(&self) -> Self::Output {
                (<$T>::one() + (-self).exp()).recip()
            }
        }
    };
}

impl_sigmoid!(f32, f64, num::Complex<f32>, num::Complex<f64>);
