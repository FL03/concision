/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, ActivateMethod, ActivationFn};
use ndarray::prelude::{Array, Dimension};
use num::One;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LinearActivation;

impl LinearActivation {
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

impl<T> ActivateMethod<T> for LinearActivation
where
    T: Clone,
{
    fn rho(&self, x: &T) -> T {
        Self::linear(x)
    }
}

impl<T, D> Activate<T, D> for LinearActivation
where
    D: Dimension,
    T: Clone,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::linear(&x))
    }
}
