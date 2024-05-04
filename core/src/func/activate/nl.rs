/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::*;
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

build_unary_trait!(ReLU.relu, Sigmoid.sigmoid, Softmax.softmax, Tanh.tanh,);

/*
 ********** Implementations **********
*/

macro_rules! impl_nl {
    ($name:ident: $($T:ty),* $(,)?) => {
        $(
            impl_nl!(@impl $name: $T);
        )*
    };
    (@impl relu: $T:ty) => {
        impl ReLU for $T {
            type Output = $T;

            fn relu(&self) -> Self::Output {
                if *self > <$T>::zero() {
                    *self
                } else {
                    <$T>::zero()
                }
            }
        }
    };
    (@impl sigmoid: $T:ty) => {
        impl Sigmoid for $T {
            type Output = $T;

            fn sigmoid(&self) -> Self::Output {
                (<$T>::one() + (-self).exp()).recip()
            }
        }
    };
    (@impl tanh: $T:ty) => {
        impl Tanh for $T {
            type Output = $T;

            fn tanh(&self) -> Self::Output {
                <$T>::tanh(*self)
            }
        }
    };
}

impl_nl!(relu: f32, f64);
impl_nl!(sigmoid: f32, f64, num::Complex<f32>, num::Complex<f64>);
impl_nl!(tanh: f32, f64, num::Complex<f32>, num::Complex<f64>);

macro_rules! impl_rho_arr {
    ($($name:ident.$call:ident),* $(,)?) => {
        $(
            impl_rho_arr!(@impl $name.$call);
        )*
    };
    (@impl $name:ident.$call:ident) => {
        impl<A, S, D> $name for ArrayBase<S, D> 
        where 
            A: Clone + $name, 
            D: Dimension, 
            S: Data<Elem = A> 
        {
            type Output = Array<<A as $name>::Output, D>;

            fn $call(&self) -> Self::Output {
                self.mapv(|x| x.$call())
            }
        }
    };
}

impl_rho_arr!(ReLU.relu, Sigmoid.sigmoid, Tanh.tanh);