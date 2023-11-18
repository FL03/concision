/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, ActivateMethod, ActivationFn};
use ndarray::prelude::{Array, Dimension};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LinearActivation;

impl LinearActivation {
    pub fn method<T>() -> ActivationFn<T> {
        |x| x
    }

    pub fn rho<T>(args: T) -> T {
        args
    }
}

impl<T> ActivateMethod<T> for LinearActivation {
    fn rho(&self, x: T) -> T {
        Self::method()(x)
    }
}

impl<T, D> Activate<T, D> for LinearActivation
where
    D: Dimension,
    T: num::Float,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| self.rho(x))
    }
}
