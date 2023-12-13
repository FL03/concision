/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::Gradient;
use ndarray::prelude::{Array, Dimension};
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(x: T) -> T
    where
        T: Float,
    {
        (-x).exp() / (T::one() + (-x).exp()).powi(2)
    }

    pub fn sigmoid<T>(x: T) -> T
    where
        T: Float,
    {
        T::one() / (T::one() + (-x).exp())
    }
}

// impl<T, D> Activate<T, D> for Sigmoid
// where
//     D: Dimension,
//     T: Float,
// {
//     fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
//         x.mapv(|x| Self::sigmoid(x))
//     }
// }

impl<T, D> Gradient<T, D> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::derivative(x))
    }
}

impl<T, D> Fn<(&Array<T, D>,)> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    extern "rust-call" fn call(&self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.mapv(|x| Self::sigmoid(x))
    }
}

impl<T, D> FnMut<(&Array<T, D>,)> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    extern "rust-call" fn call_mut(&mut self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.mapv(|x| Self::sigmoid(x))
    }
}

impl<T, D> FnOnce<(&Array<T, D>,)> for Sigmoid
where
    D: Dimension,
    T: Float,
{
    type Output = Array<T, D>;

    extern "rust-call" fn call_once(self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.mapv(|x| Self::sigmoid(x))
    }
}
