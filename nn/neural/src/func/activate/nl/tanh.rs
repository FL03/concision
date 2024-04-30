/*
    Appellation: tanh <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::Gradient;
use ndarray::prelude::{Array, Dimension};
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(args: &T) -> T
    where
        T: Float,
    {
        T::one() - args.tanh().powi(2)
    }

    pub fn tanh<T>(args: &T) -> T
    where
        T: Float,
    {
        args.tanh()
    }
}

// impl<T, D> Activate<T, D> for Tanh
// where
//     D: Dimension,
//     T: Float,
// {
//     fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
//         x.mapv(Float::tanh)
//     }
// }

impl<T, D> Gradient<T, D> for Tanh
where
    D: Dimension,
    T: Float,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::derivative(&x))
    }
}

impl<T, D> Fn<(&Array<T, D>,)> for Tanh
where
    D: Dimension,
    T: Float,
{
    extern "rust-call" fn call(&self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.mapv(T::tanh)
    }
}

impl<T, D> FnMut<(&Array<T, D>,)> for Tanh
where
    D: Dimension,
    T: Float,
{
    extern "rust-call" fn call_mut(&mut self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.mapv(T::tanh)
    }
}

impl<T, D> FnOnce<(&Array<T, D>,)> for Tanh
where
    D: Dimension,
    T: Float,
{
    type Output = Array<T, D>;

    extern "rust-call" fn call_once(self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.mapv(T::tanh)
    }
}
