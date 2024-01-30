/*
    Appellation: relu <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::Gradient;
use ndarray::prelude::{Array, Dimension};
use num::{One, Zero};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn derivative<T>(args: &T) -> T
    where
        T: One + PartialOrd + Zero,
    {
        if args > &T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }

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
}

// impl<T, D> Activate<T, D> for ReLU
// where
//     D: Dimension,
//     T: Clone + PartialOrd + Zero,
// {
//     fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
//         x.map(Self::relu)
//     }
// }

impl<T, D> Gradient<T, D> for ReLU
where
    D: Dimension,
    T: Clone + One + PartialOrd + Zero,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        args.map(Self::derivative)
    }
}

impl<T, D> Fn<(&Array<T, D>,)> for ReLU
where
    D: Dimension,
    T: Clone + PartialOrd + Zero,
{
    extern "rust-call" fn call(&self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.map(Self::relu)
    }
}

impl<T, D> FnMut<(&Array<T, D>,)> for ReLU
where
    D: Dimension,
    T: Clone + PartialOrd + Zero,
{
    extern "rust-call" fn call_mut(&mut self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.map(Self::relu)
    }
}

impl<T, D> FnOnce<(&Array<T, D>,)> for ReLU
where
    D: Dimension,
    T: Clone + PartialOrd + Zero,
{
    type Output = Array<T, D>;

    extern "rust-call" fn call_once(self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.map(Self::relu)
    }
}
