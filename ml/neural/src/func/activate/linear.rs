/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, ActivationFn};
use ndarray::prelude::{Array, Dimension};
use num::One;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Linear;

impl Linear {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn derivative<T>(_x: T) -> T
    where
        T: One,
    {
        T::one()
    }

    pub fn gradient<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Clone + One,
    {
        args.mapv(|x| Self::derivative(x))
    }

    pub fn linear<T: Clone>(args: &T) -> T {
        args.clone()
    }

    pub fn method<T>() -> ActivationFn<T> {
        |x| x
    }

    pub fn rho<T>(args: T) -> T {
        args
    }
}

impl<T, D> Activate<T, D> for Linear
where
    D: Dimension,
    T: Clone,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        args.clone()
    }
}
